import sys
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
import pickle
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rich import print as rp
import lmdb

from util.constants import AA_NAME_SYM, BACKBONE_NAMES
from util.tools import set_seed
from util.data_load import SequenceData
from util.seq_process import read_seq_data, SEQ_LMDB_CONFIG
from util.seq_process import connect_lmdb as connect_seq_lmdb


def _parse_ec_number(ec_str, caps=(8, 200, 200, 1200)):
    """Parse EC number string like '1.2.3.4' into 4 integer indices.

    Returns [ec1, ec2, ec3, ec4] where 0 means unknown/wildcard.
    Values are clamped to [1, cap-1] to stay within embedding vocab.
    """
    try:
        parts = str(ec_str).replace(' ', '').split('.')
    except Exception:
        return [0, 0, 0, 0]

    result = []
    for i in range(4):
        if i < len(parts):
            try:
                val = int(parts[i])
                val = min(max(1, val), caps[i] - 1)
            except (ValueError, TypeError):
                val = 0  # wildcard '-' or missing
        else:
            val = 0
        result.append(val)
    return result


class SeqMolComplexData(torch_geometric.data.Data):
    """
    Sequence + Molecular features data point (no structure)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_dict(sequence=None, molecular=None, **kwargs):
        instance = SeqMolComplexData(**kwargs)

        if sequence is not None:
            for key, item in sequence.items():
                instance['SEQ_' + key] = item

        if molecular is not None:
            for key, item in molecular.items():
                instance['MOL_' + key] = item

        # ���� num_nodes - �����������ݣ���Ϊ���г���
        if hasattr(instance, 'SEQ_seq_padding_mask'):
            if instance.SEQ_seq_padding_mask.dim() == 1:
                instance.num_nodes = instance.SEQ_seq_padding_mask.shape[0]
            else:
                instance.num_nodes = instance.SEQ_seq_padding_mask.shape[1]
        else:
            instance.num_nodes = 1

        return instance

    def __inc__(self, key, value, *args, **kwargs):
        # ������������Ҫ����������
        if key in ['MOL_morgan', 'MOL_grover_mean', 'MOL_molt5']:
            return 0
        # EC number indices — no offset needed
        elif key == 'EC_ids':
            return 0
        # �����������Ҳ����Ҫ����������
        elif key.startswith('SEQ_'):
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)


class SeqMolRepresenter(torch.utils.data.Dataset):
    """
    Dataset for sequence + molecular features (no structure)
    """
    def __init__(self, config, df: pd.DataFrame, is_train: bool = False) -> None:
        super().__init__()

        self.config = config
        self.df = df
        self.is_train = is_train
        
        # Validate dataframe columns
        required_cols = ['DATA_ID', 'SEQ_ID', 'SMI_ID', 'EC_NUMBER', 'Y_VALUE']
        assert all(c in df.columns for c in required_cols), \
            f"Dataframe must contain {required_cols}, got {df.columns.tolist()}"
        
        self.data_ids: List[str] = self.df.loc[:, 'DATA_ID'].tolist()
        self.seq_ids: List[str] = self.df.loc[:, 'SEQ_ID'].tolist()
        self.smi_ids: List[str] = self.df.loc[:, 'SMI_ID'].tolist()
        self.ec_s: List[str]   = self.df.loc[:, 'EC_NUMBER'].tolist()
        self.y_s = self.df.loc[:, 'Y_VALUE'].tolist()

        self.use_ec = config['model'].get('use_ec', False)
        
        # Sequence LMDB configuration
        self.seq_lmdb_config = SEQ_LMDB_CONFIG(
            seq_fp=self.config["data"]["seq_lmdb"]["seq_fp"],
            lmdb_fp=self.config["data"]["seq_lmdb"]["lmdb"],
            map_size=self.config["data"]["seq_lmdb"]["map_size"],
            max_seq_len=self.config["data"]["seq_lmdb"]["max_seq_len"]
        )
        # �ӳټ�����������
        self.seq_db = None

        # Molecular features configuration
        self.use_morgan = config["model"]["embedding"].get("morgan", False)
        self.use_grover = config["model"]["embedding"].get("grover", False)
        
        # Load Morgan fingerprints (numpy array is safe for multiprocessing)
        self.morgan_fps = None
        if self.use_morgan:
            morgan_path = config["data"].get("morgan_path")
            if morgan_path and os.path.exists(morgan_path):
                self.morgan_fps = np.load(morgan_path).astype(np.float32)
                print(f"? Loaded Morgan fingerprints: {self.morgan_fps.shape}")
            else:
                print(f"? Warning: Morgan path {morgan_path} not found")
                self.use_morgan = False

        # Load MolT5 embeddings (numpy array, same layout as Morgan)
        self.use_molt5 = config['model']['embedding'].get('molt5', False)
        self.molt5_fps = None
        if self.use_molt5:
            molt5_path = config["data"].get("molt5_path")
            if molt5_path and os.path.exists(molt5_path):
                self.molt5_fps = np.load(molt5_path).astype(np.float32)
                print(f"Loaded MolT5 embeddings: {self.molt5_fps.shape}")
            else:
                print(f"Warning: MolT5 path {molt5_path} not found")
                self.use_molt5 = False

        # GROVER LMDB - only store path, connect lazily
        self.grover_path = None
        self.grover_db = None
        if self.use_grover:
            self.grover_path = config["data"].get("grover_path")
            if not (self.grover_path and os.path.exists(self.grover_path)):
                print(f"Warning: GROVER path {self.grover_path} not found")
                self.use_grover = False


    def _get_seq_db(self):
        """Lazy initialization of sequence LMDB connection (per worker)"""
        if self.seq_db is None:
            self.seq_db = connect_seq_lmdb(self.seq_lmdb_config)
        return self.seq_db

    def _get_grover_db(self):
        """Lazy initialization of GROVER LMDB connection (per worker)"""
        if self.grover_db is None and self.grover_path is not None:
            try:
                self.grover_db = lmdb.open(
                    self.grover_path,
                    map_size=2 * 1024 ** 3,  # 2GB — grover_fingerprint.lmdb is ~660MB
                    create=False,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=1024,
                )
            except Exception as e:
                print(f"Warning: Could not open GROVER LMDB: {e}")
                self.use_grover = False
        return self.grover_db

    def __del__(self):
        """Clean up database connections"""
        if hasattr(self, 'seq_db') and self.seq_db is not None:
            self.seq_db.close()
        if hasattr(self, 'grover_db') and self.grover_db is not None:
            self.grover_db.close()

    def __len__(self):
        return len(self.data_ids)

    def _get_sequence_data(self, seq_id: str):
        """Load sequence data from LMDB"""
        db = self._get_seq_db()
        try:
            sequence_data = read_seq_data(db, seq_id)
        except KeyError:
            print(f"Warning: Sequence '{seq_id}' not found in LMDB, using zero embedding")
            max_len = self.seq_lmdb_config.max_seq_len
            from util.data_load import SequenceData, padding_seq_embedding
            dat = {"embedding": np.zeros((1, 1024), dtype=np.float32), "sequence": "X"}
            dat = padding_seq_embedding(dat, max_len)
            sequence_data = SequenceData.from_dict(dat)
        # Force float32 (handles bf16/fp16 embeddings from ProtT5 etc.)
        emb = sequence_data.embedding
        if isinstance(emb, torch.Tensor):
            sequence_data.embedding = emb.float()
        else:
            sequence_data.embedding = torch.as_tensor(emb, dtype=torch.float32)
        sequence_data.seq_padding_mask = torch.tensor(
            sequence_data.seq_padding_mask,
            dtype=torch.bool           
        )
        return sequence_data

    def _parse_smi_idx(self, smi_id: str) -> int:
        """Extract numeric index from SMI_ID string."""
        try:
            if smi_id.startswith('SMI'):
                return int(smi_id.replace('SMI', ''))
            return int(smi_id)
        except Exception:
            print(f"Warning: Could not parse SMI_ID: {smi_id}, using index 0")
            return 0

    def _load_numpy_feature(self, array, smi_idx: int, name: str):
        """Load a feature from a preloaded numpy array by index."""
        if array is not None and smi_idx < len(array):
            return torch.from_numpy(array[smi_idx])
        return None

    def _load_grover_feature(self, smi_idx: int):
        """Load GROVER features from LMDB."""
        db = self._get_grover_db()
        if db is None:
            return None
        try:
            with db.begin(write=False) as txn:
                value = txn.get(str(smi_idx).encode())
                if value is not None:
                    grover_data = pickle.loads(value)
                    return torch.from_numpy(grover_data['total_embedding']).float()
        except Exception as e:
            print(f"Warning: Could not load GROVER features for index {smi_idx}: {e}")
        return None

    def __getitem__(self, idx: int) -> SeqMolComplexData:
        seq_id = self.seq_ids[idx]
        smi_idx = self._parse_smi_idx(self.smi_ids[idx])

        seq_data = self._get_sequence_data(seq_id)

        # Molecular features
        mol_features = {}
        if self.use_morgan:
            feat = self._load_numpy_feature(self.morgan_fps, smi_idx, 'morgan')
            morgan_dim = self.config['model']['embedding'].get('morgan_dim', 1024)
            mol_features['morgan'] = feat if feat is not None else torch.zeros(morgan_dim)
        if self.use_molt5:
            feat = self._load_numpy_feature(self.molt5_fps, smi_idx, 'molt5')
            molt5_dim = self.config['model']['embedding'].get('molt5_dim', 768)
            mol_features['molt5'] = feat if feat is not None else torch.zeros(molt5_dim)
        if self.use_grover:
            feat = self._load_grover_feature(smi_idx)
            grover_dim = self.config['model']['embedding'].get('grover_dim', 4885)
            mol_features['grover_mean'] = feat if feat is not None else torch.zeros(grover_dim)

        # Combine
        complex_data = SeqMolComplexData.from_dict(
            sequence=seq_data, molecular=mol_features
        )
        if self.use_ec:
            ec_ids = _parse_ec_number(self.ec_s[idx])
            complex_data.EC_ids = torch.tensor([ec_ids], dtype=torch.long)
        complex_data.y = torch.tensor([self.y_s[idx]], dtype=torch.float)

        # Store SEQ_ID and SMI_ID for ranking losses (string, not tensor)
        complex_data.SEQ_seq_id = seq_id
        complex_data.MOL_smi_id = self.smi_ids[idx]

        return complex_data


class seq_representer(torch.utils.data.Dataset):
    """
    Dataset for sequence data only (kept for compatibility)
    """
    def __init__(self, config, seq_id: List[str]) -> None:
        super().__init__()
        self.config = config
        self.seq_ids = seq_id
        self.lmdb_config = SEQ_LMDB_CONFIG(
            seq_fp=self.config["data"]["seq_lmdb"]["seq_fp"],
            lmdb_fp=self.config["data"]["seq_lmdb"]["lmdb"],
            map_size=self.config["data"]["seq_lmdb"]["map_size"],
            max_seq_len=self.config["data"]["seq_lmdb"]["max_seq_len"]
        )
        self.db = None

    def _get_db(self):
        if self.db is None:
            self.db = connect_seq_lmdb(self.lmdb_config)
        return self.db

    def __del__(self):
        if hasattr(self, 'db') and self.db is not None:
            self.db.close()

    def __len__(self):
        return len(self.seq_ids)
    
    def get_item_from_id(self, db, seq_id: str) -> SequenceData:
        sequence_data = read_seq_data(db, seq_id)
        # Force float32 (handles bf16/fp16 embeddings from ProtT5 etc.)
        emb = sequence_data.embedding
        if isinstance(emb, torch.Tensor):
            sequence_data.embedding = emb.float()
        else:
            sequence_data.embedding = torch.as_tensor(emb, dtype=torch.float32)
        sequence_data.seq_padding_mask = torch.tensor(
            sequence_data.seq_padding_mask,
            dtype=torch.bool           
        )
        return sequence_data

    def __getitem__(self, idx: int) -> SequenceData:
        seq_id = self.seq_ids[idx]
        db = self._get_db()
        return self.get_item_from_id(db, seq_id)


class GroupedSampler(torch.utils.data.Sampler):
    """
    Batch sampler that groups samples by a given column so related samples
    are likely in the same batch — enables effective pairwise ranking loss.
    """
    def __init__(self, df: pd.DataFrame, group_col: str, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        from collections import defaultdict
        self.groups = defaultdict(list)
        for idx, gid in enumerate(df[group_col]):
            self.groups[gid].append(idx)
        self.group_keys = list(self.groups.keys())

    def __iter__(self):
        import random
        if self.shuffle:
            random.shuffle(self.group_keys)

        batch = []
        for key in self.group_keys:
            indices = self.groups[key]
            if self.shuffle:
                random.shuffle(indices)
            batch.extend(indices)
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        if batch:
            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.groups.values())
        return (total + self.batch_size - 1) // self.batch_size


# Keep old name for backward compatibility
EnzymeGroupedSampler = GroupedSampler


class Singledataset(pl.LightningDataModule):
    """
    Lightning DataModule for sequence + molecular features
    """
    def __init__(self, config):
        super().__init__()

        set_seed(config["train"]["seed"])
        self.config = config
        self.batch_size = config["train"]["batch_size"]
        self.n_worker = config["train"]["n_cpus"]

        # Load dataframes (raw Y_VALUE = kcat/Km, log10 transform done in model)
        self.train_df = pd.read_csv(config["data"]["train_data_df"])
        self.eval_df = pd.read_csv(config["data"]["eval_data_df"])

        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_df)} samples")
        print(f"  Eval: {len(self.eval_df)} samples")

        # Cache datasets so reload_dataloaders_every_n_epochs doesn't re-load data
        self._train_data = get_representer(config=self.config, df=self.train_df, is_train=True)
        self._eval_data = get_representer(config=self.config, df=self.eval_df, is_train=False)

    def train_dataloader(self):
        data = self._train_data
        use_enz_rank = self.config['model'].get('rank_loss_weight', 0.0) > 0
        use_sub_rank = self.config['model'].get('rank_loss_substrate_weight', 0.0) > 0

        if use_enz_rank or use_sub_rank:
            # Alternate grouping: even epochs → SEQ_ID, odd epochs → SMI_ID
            epoch = self.trainer.current_epoch if self.trainer else 0
            if use_enz_rank and use_sub_rank:
                group_col = 'SEQ_ID' if epoch % 2 == 0 else 'SMI_ID'
            elif use_enz_rank:
                group_col = 'SEQ_ID'
            else:
                group_col = 'SMI_ID'
            sampler = GroupedSampler(self.train_df, group_col, self.batch_size, shuffle=True)
            return DataLoader(
                data,
                batch_sampler=sampler,
                num_workers=self.n_worker,
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=4 if self.n_worker > 0 else None
            )
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_worker,
            persistent_workers=False if self.n_worker == 0 else True,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=4 if self.n_worker > 0 else None
        )

    def val_dataloader(self):
        return DataLoader(
            self._eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_worker,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=4 if self.n_worker > 0 else None
        )

    def test_dataloader(self):
        return DataLoader(
            self._eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_worker,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=4 if self.n_worker > 0 else None
        )


def get_representer(config, df, is_train=True):
    """
    Factory function to get the appropriate representer
    """
    representer_type = config["data"]["data_representer"]
    
    if representer_type == 'sequence_molecular':
        return SeqMolRepresenter(config=config, df=df, is_train=is_train)
    else:
        raise ValueError(f"Unknown representer type: {representer_type}")