#!/usr/bin/env python
"""
DLCatalysis inference script.
Predicts enzyme catalytic efficiency (kcat/Km, M^-1 s^-1) given enzyme
amino acid sequence(s) and substrate SMILES.

Embedding modes:
  - LoRA (use_lora=true): loads ProtT5 weights, runs forward pass in realtime
  - Frozen + LMDB: reads precomputed embeddings from seq_lmdb
  - Frozen + no LMDB: loads ProtT5 weights on-the-fly for inference

Usage:
    # Single prediction
    python run_model.py \
        --config ../../config/test.yml \
        --ckpt  ../../train/hpo_ckt/best.ckpt \
        --seq   MKTAYIAK... \
        --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
        --ec    "3.1.1.1"

    # Batch prediction from CSV
    python run_model.py \
        --config ../../config/test.yml \
        --ckpt  ../../train/hpo_ckt/best.ckpt \
        --input  batch_input.csv \
        --output predictions.csv

    # batch_input.csv format:
    # seq_id,smiles,sequence[,ec_number]
    # enz1,CC(=O)O,MKTAYI...,3.1.1.1
"""
import sys
import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model.enzsub import EnzSub
from util.tools import init_config
from util.data_module import SeqMolComplexData, _parse_ec_number


# ── ProtT5 dimension ──────────────────────────────────────────────────────────
_PROT5_DIM = 1024

def _get_seq_dim(config: dict) -> int:
    return _PROT5_DIM


# ── Sequence embedding provider ─────────────────────────────────────────────
class EmbeddingProvider:
    """Provides sequence embeddings via LMDB lookup or realtime ProtT5 forward."""

    def __init__(self, config: dict, device: str = 'cpu', use_lmdb: bool = False):
        self.config = config
        self.device = device
        self.mode = None  # 'lmdb', 'realtime', or 'lora'
        self._lmdb_env = None
        self._seq_encoder = None
        self._max_len = config['data']['seq_lmdb']['max_seq_len']
        self._seq_dim = _get_seq_dim(config)

        use_lora = config['model'].get('use_lora', False)

        if use_lora:
            # LoRA mode: model handles embedding internally via forward_batch
            self.mode = 'lora'
            print(f"[EmbeddingProvider] LoRA mode — model runs ProtT5 forward internally")
            return

        # Frozen mode: default to realtime ProtT5 forward for inference
        # Only use LMDB when explicitly requested (e.g. benchmark evaluation)
        if use_lmdb:
            lmdb_path = config['data']['seq_lmdb'].get('lmdb', '')
            if lmdb_path and os.path.exists(lmdb_path):
                self.mode = 'lmdb'
                print(f"[EmbeddingProvider] LMDB mode — reading from {lmdb_path}")
                return
            print(f"[EmbeddingProvider] LMDB requested but not found, falling back to realtime")

        self.mode = 'realtime'
        print(f"[EmbeddingProvider] Realtime mode — loading ProtT5 for on-the-fly embedding")
        self._init_seq_encoder()

    def _init_seq_encoder(self):
        from util.featurize.seq_esm2 import esm_embedding
        encoder_config = dict(self.config['model'])
        encoder_config['precomputed_only'] = False  # force loading weights
        self._seq_encoder = esm_embedding(
            device=torch.device(self.device), config=encoder_config
        )
        self._seq_encoder.eval()
        for p in self._seq_encoder.parameters():
            p.requires_grad = False

    def _get_lmdb_env(self):
        if self._lmdb_env is None:
            import lmdb
            lmdb_path = self.config['data']['seq_lmdb']['lmdb']
            self._lmdb_env = lmdb.open(
                lmdb_path, readonly=True, lock=False, subdir=False,
                map_size=self.config['data']['seq_lmdb'].get('map_size', 600 * 1024**3),
                max_readers=1024,
            )
        return self._lmdb_env

    def get_embedding(self, sequence: str, seq_id: str = None) -> torch.Tensor:
        """Returns (max_len, seq_dim) float32 tensor."""
        max_len = self._max_len
        seq = sequence[:max_len]
        seq_len = len(seq)

        if self.mode == 'lora':
            # LoRA mode: return zeros, model will run ProtT5 forward internally
            return torch.zeros(max_len, self._seq_dim, dtype=torch.float32)

        if self.mode == 'lmdb' and seq_id is not None:
            # Try LMDB lookup
            try:
                from util.seq_process import read_seq_data
                env = self._get_lmdb_env()
                seq_data = read_seq_data(env, seq_id)
                emb = seq_data.embedding
                if isinstance(emb, torch.Tensor):
                    emb = emb.float()
                else:
                    emb = torch.as_tensor(emb, dtype=torch.float32)
                # Pad/truncate to max_len
                if emb.shape[0] < max_len:
                    pad = torch.zeros(max_len - emb.shape[0], emb.shape[1])
                    emb = torch.cat([emb, pad], dim=0)
                else:
                    emb = emb[:max_len]
                return emb
            except Exception as e:
                print(f"Warning: LMDB lookup failed for {seq_id}: {e}, falling back to realtime")

        # Realtime ProtT5 forward — use last_hidden_state in fp16 (matches LMDB precomputed)
        if self._seq_encoder is None:
            self._init_seq_encoder()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            input_ids, attention_mask = self._seq_encoder._tokenize([seq])
            device = next(self._seq_encoder.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = self._seq_encoder.model(input_ids=input_ids, attention_mask=attention_mask)
            token_rep = outputs.last_hidden_state  # (1, L, 1024) — after final_layer_norm
            seq_len = min(len(seq), int(attention_mask[0].sum().item()) - 1)
            emb = token_rep[0, :seq_len]  # (seq_len, 1024)
        emb = emb.cpu().float()
        if emb.shape[0] < max_len:
            pad = torch.zeros(max_len - emb.shape[0], emb.shape[1])
            emb = torch.cat([emb, pad], dim=0)
        else:
            emb = emb[:max_len]
        return emb

    def close(self):
        if self._lmdb_env is not None:
            self._lmdb_env.close()


# ── MolT5 embedding ──────────────────────────────────────────────────────────
_molt5_model = None
_molt5_tokenizer = None

def smiles_to_molt5(smiles: str, molt5_dim: int = 768, molt5_path: str = None) -> np.ndarray:
    global _molt5_model, _molt5_tokenizer
    try:
        from transformers import T5EncoderModel, T5Tokenizer
        if _molt5_model is None:
            if molt5_path:
                model_name = molt5_path
            elif molt5_dim == 512:
                model_name = "laituan245/molt5-small"
            elif molt5_dim == 1024:
                model_name = "laituan245/molt5-large"
            else:
                model_name = "laituan245/molt5-base"
            _molt5_tokenizer = T5Tokenizer.from_pretrained(model_name)
            _molt5_model = T5EncoderModel.from_pretrained(model_name)
            _molt5_model.eval()
        with torch.no_grad():
            inputs = _molt5_tokenizer(smiles, return_tensors="pt",
                                      padding=True, truncation=True, max_length=512)
            outputs = _molt5_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return emb.astype(np.float32)
    except Exception as e:
        print(f"Warning: MolT5 encoding failed: {e}")
        return np.zeros(molt5_dim, dtype=np.float32)


# ── Morgan fingerprint (RDKit) ───────────────────────────────────────────────
def smiles_to_morgan(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: invalid SMILES '{smiles}', using zero fingerprint")
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except ImportError:
        print("Warning: RDKit not installed, Morgan fingerprint will be zeros")
        return np.zeros(n_bits, dtype=np.float32)


# ── SMILES desalting (consistent with brenda_tools.py) ───────────────────────
def _desalt_smiles(smi: str) -> str:
    """Remove counter-ions but keep physiological charges."""
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi


# ── Build a single Data object ───────────────────────────────────────────────
def build_data(sequence: str, smiles: str, config: dict,
               embedding: torch.Tensor,
               max_len: int = 1000, ec_number: str = None,
               msa_provider=None, seq_id: str = None) -> SeqMolComplexData:
    smiles = _desalt_smiles(smiles)
    seq = sequence[:max_len]
    seq_len = len(seq)

    # Padding mask: True = padding position
    padding_mask = np.ones((1, max_len), dtype=bool)
    padding_mask[0, :seq_len] = False

    sequence_dict = {
        'sequence':         seq,
        'seq_padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
        'embedding':        embedding.unsqueeze(0),  # (1, max_len, seq_dim)
    }

    mol_dict = {}
    cfg_emb = config['model']['embedding']

    if cfg_emb.get('morgan', False):
        n_bits = cfg_emb.get('morgan_dim', 1024)
        mol_dict['morgan'] = torch.from_numpy(smiles_to_morgan(smiles, n_bits=n_bits))

    if cfg_emb.get('molt5', False):
        molt5_dim = cfg_emb.get('molt5_dim', 768)
        molt5_path = config['model'].get('molt5_model_path', None)
        mol_dict['molt5'] = torch.from_numpy(smiles_to_molt5(smiles, molt5_dim, molt5_path=molt5_path))

    if cfg_emb.get('grover', False):
        grover_dim = cfg_emb.get('grover_dim', 4885)
        if not getattr(build_data, '_grover_warned', False):
            print("Warning: GROVER features require precomputation, using zeros for inference")
            build_data._grover_warned = True
        mol_dict['grover_mean'] = torch.zeros(grover_dim, dtype=torch.float32)

    data = SeqMolComplexData.from_dict(sequence=sequence_dict, molecular=mol_dict)

    if config['model'].get('use_ec', False):
        ec_ids = _parse_ec_number(ec_number) if ec_number else [0, 0, 0, 0]
        data.EC_ids = torch.tensor([ec_ids], dtype=torch.long)

    # MSA residue-level features (optional)
    if config['model'].get('use_msa', False):
        msa_dim = config['model'].get('msa_dim', 768)
        msa_residue = None
        log_neff = None
        if msa_provider is not None and seq_id is not None:
            msa_residue, log_neff = msa_provider.get_msa(seq_id, max_len)
        if msa_residue is None:
            msa_residue = torch.zeros(max_len, msa_dim, dtype=torch.float32)
            log_neff = torch.zeros(1, dtype=torch.float32)
        data.MSA_residue = msa_residue
        data.MSA_log_neff = log_neff

    data.y = torch.tensor([0.0], dtype=torch.float)
    return data


# ── MSA feature provider ─────────────────────────────────────────────────────
class MSAProvider:
    """Provides MSA residue-level features from precomputed LMDB."""

    def __init__(self, lmdb_path: str):
        self._env = None
        self._path = lmdb_path

    def _get_env(self):
        if self._env is None:
            import lmdb
            self._env = lmdb.open(
                self._path, readonly=True, lock=False, subdir=False,
                map_size=50 * 1024**3, max_readers=1024,
            )
        return self._env

    def get_msa(self, seq_id: str, max_len: int = 1000):
        """Returns (max_len, 768) residue tensor + (1,) log_neff, or (None, None)."""
        try:
            env = self._get_env()
            with env.begin(write=False) as txn:
                raw = txn.get(seq_id.encode())
                if raw is None:
                    return None, None
                payload = pickle.loads(raw)
            emb = payload['embedding']  # (L, 768) float16
            if emb.shape[0] > max_len:
                emb = emb[:max_len]
            L = emb.shape[0]
            residue_emb = np.zeros((max_len, emb.shape[1]), dtype=np.float32)
            residue_emb[:L] = emb.astype(np.float32)
            residue_tensor = torch.from_numpy(residue_emb)
            log_neff = torch.tensor([payload.get('log_neff', 0.0)], dtype=torch.float32)
            return residue_tensor, log_neff
        except Exception as e:
            print(f"Warning: MSA load failed for {seq_id}: {e}")
            return None, None

    def close(self):
        if self._env is not None:
            self._env.close()


# ── Main inference function ──────────────────────────────────────────────────
@torch.no_grad()
def predict(model: EnzSub, sequences: list, smiles_list: list,
            config: dict, device: str = 'cuda',
            seq_ids: list = None, ec_numbers: list = None,
            batch_size: int = 4, use_lmdb: bool = False) -> np.ndarray:
    """
    Predict kcat/Km for a list of (sequence, smiles) pairs.
    Returns predicted values in log10 space (log10(kcat/Km)).
    """
    n = len(sequences)
    if len(smiles_list) != n:
        raise ValueError(f"sequences ({n}) and smiles_list ({len(smiles_list)}) must have same length")

    model.eval()
    model = model.to(device)
    max_len = config['data']['seq_lmdb']['max_seq_len']

    if seq_ids is None:
        seq_ids = [f'query_{i}' for i in range(n)]
    if ec_numbers is None:
        ec_numbers = [None] * n
    if len(seq_ids) != n or len(ec_numbers) != n:
        raise ValueError("seq_ids and ec_numbers must match sequences length")

    emb_provider = EmbeddingProvider(config, device=device, use_lmdb=use_lmdb)

    all_preds = []
    try:
        for i in range(0, n, batch_size):
            batch_seqs   = sequences[i:i + batch_size]
            batch_smiles = smiles_list[i:i + batch_size]
            batch_ids    = seq_ids[i:i + batch_size]
            batch_ecs    = ec_numbers[i:i + batch_size]

            data_list = []
            for seq, smi, sid, ec in zip(batch_seqs, batch_smiles, batch_ids, batch_ecs):
                emb = emb_provider.get_embedding(seq, seq_id=sid)
                data_list.append(
                    build_data(seq, smi, config, emb, max_len=max_len, ec_number=ec)
                )
            batch = Batch.from_data_list(data_list).to(device)

            y_pred, _ = model(batch)
            raw_pred = y_pred.squeeze(-1).cpu().numpy()
            all_preds.extend(raw_pred.tolist())
    finally:
        emb_provider.close()

    return np.array(all_preds)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='DLCatalysis kcat/Km prediction')
    parser.add_argument('--config', type=str, required=True,  help='Path to config YAML')
    parser.add_argument('--ckpt',   type=str, default=None,  help='Path to single checkpoint .ckpt')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Path to checkpoint dir with fold0/~fold9/ subdirs for 10-fold ensemble')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_lmdb', action='store_true',
                        help='Use precomputed LMDB embeddings (for benchmark evaluation)')

    # Single prediction mode
    parser.add_argument('--seq',    type=str, default=None, help='Enzyme amino acid sequence')
    parser.add_argument('--smiles', type=str, default=None, help='Substrate SMILES')
    parser.add_argument('--ec',     type=str, default=None, help='EC number (e.g. 1.2.3.4)')

    # Batch prediction mode
    parser.add_argument('--input',  type=str, default=None,
                        help='CSV with columns: seq_id, smiles, sequence [, ec_number]')
    parser.add_argument('--output', type=str, default='predictions.xlsx')
    args = parser.parse_args()

    if not args.ckpt and not args.ckpt_dir:
        parser.error("Must provide --ckpt or --ckpt_dir")

    config = init_config(args.config)
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Collect checkpoint paths
    if args.ckpt_dir:
        import glob, re
        ckpt_paths = []

        # Try flat format first: fold0_*.ckpt, fold1_*.ckpt, ...
        flat_ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "fold*.ckpt")))
        if flat_ckpts:
            ckpt_paths = [f for f in flat_ckpts if "last" not in os.path.basename(f)]
        else:
            # Try fold subdirectories: fold0/, fold1/, ...
            for fold in range(10):
                fold_dir = os.path.join(args.ckpt_dir, f"fold{fold}")
                if not os.path.exists(fold_dir):
                    continue
                ckpts = [f for f in glob.glob(os.path.join(fold_dir, "**", "*.ckpt"), recursive=True)
                         if "last" not in os.path.basename(f)]
                if not ckpts:
                    ckpts = [f for f in glob.glob(os.path.join(fold_dir, "*.ckpt"))
                             if "last" not in os.path.basename(f)]
                if len(ckpts) > 1:
                    def _extract_mse(path):
                        m = re.search(r'MSE=([\d]+[.]\d+)', path)
                        return float(m.group(1)) if m else 999
                    if any('MSE=' in c for c in ckpts):
                        ckpts = [min(ckpts, key=_extract_mse)]
                    else:
                        ckpts = [max(ckpts, key=os.path.getmtime)]
                if ckpts:
                    ckpt_paths.append(ckpts[0])

        print(f"Found {len(ckpt_paths)} fold checkpoints for ensemble")
    else:
        ckpt_paths = [args.ckpt]

    if args.seq and args.smiles:
        # Single prediction with ensemble
        all_preds = []
        for i, ckpt in enumerate(ckpt_paths):
            model = EnzSub.load_from_checkpoint(ckpt, config=config, map_location=device)
            model.eval()
            log10_val = predict(model, [args.seq], [args.smiles], config,
                                device=device, ec_numbers=[args.ec], batch_size=1,
                                use_lmdb=args.use_lmdb)
            all_preds.append(log10_val[0])
            del model
            torch.cuda.empty_cache()

        mean_pred = np.mean(all_preds)
        kcat_km = 10 ** mean_pred
        print(f"\nEnsemble ({len(all_preds)} folds):")
        print(f"  Predicted kcat/Km: {kcat_km:.4e} M-1 s-1  (log10: {mean_pred:.4f})")

    elif args.input:
        # Batch prediction with ensemble
        if args.input.endswith('.xlsx'):
            df = pd.read_excel(args.input, engine='openpyxl')
        else:
            df = pd.read_csv(args.input)
        if 'sequence' not in df.columns or 'smiles' not in df.columns:
            raise ValueError("CSV must have 'sequence' and 'smiles' columns")

        seq_ids   = df.get('seq_id', pd.Series([f'query_{i}' for i in range(len(df))])).tolist()
        sequences = df['sequence'].tolist()
        smiles    = df['smiles'].tolist()
        ec_numbers = df['ec_number'].tolist() if 'ec_number' in df.columns else [None] * len(df)

        all_fold_preds = []
        for i, ckpt in enumerate(ckpt_paths):
            print(f"Loading fold {i} from {ckpt} ...")
            model = EnzSub.load_from_checkpoint(ckpt, config=config, map_location=device)
            model.eval()
            log10_vals = predict(model, sequences, smiles, config,
                                 device=device, seq_ids=seq_ids, ec_numbers=ec_numbers,
                                 batch_size=args.batch_size, use_lmdb=args.use_lmdb)
            all_fold_preds.append(log10_vals)
            del model
            import gc; gc.collect()
            torch.cuda.empty_cache()
            print(f"  Fold {i} done")

        # Average across folds
        ensemble_preds = np.mean(all_fold_preds, axis=0)
        df['predicted_log10_kcat_km'] = ensemble_preds
        df['predicted_kcat_km'] = 10 ** ensemble_preds
        out_cols = ['seq_id', 'smiles', 'ec_number', 'predicted_log10_kcat_km', 'predicted_kcat_km']
        out_cols = [c for c in out_cols if c in df.columns]
        out_df = df[out_cols]
        if args.output.endswith('.xlsx'):
            out_df.to_excel(args.output, index=False, engine='openpyxl')
        else:
            out_df.to_csv(args.output, sep='\t', index=False)
        print(f"\nEnsemble predictions ({len(ckpt_paths)} folds) saved to {args.output}")
        print(df[['seq_id', 'predicted_log10_kcat_km', 'predicted_kcat_km']].head(10).to_string(index=False))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
