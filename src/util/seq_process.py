import lmdb
import pandas as pd
from rich import print as rp
from rich.progress import track
from torch import device
import pickle
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(PROJECT_ROOT)
from util.data_load import SequenceData, padding_seq_embedding

from typing import List, Tuple, Any, Dict
from dataclasses import dataclass

@dataclass
class SEQ_LMDB_CONFIG:
    seq_fp: str
    lmdb_fp: str
    map_size: int = 600*(1024*1024*1024)  # 600 GB
    max_seq_len: int = 1000               # 最大序列长度限制, 后面会填充mask

def init_lmdb(seq_config) -> Any:
    env = lmdb.open(
        seq_config.lmdb_fp,
        map_size = seq_config.map_size, 
        create = True,
        subdir = False,
        readonly = False,  
    )

    return env

def connect_lmdb(seq_config):
    """
    这个函数只用来打开一个已经存在的db
    """
    mydb = lmdb.open(
        seq_config.lmdb_fp,
        map_size = seq_config.map_size,
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1024,
    )

    return mydb

def write_seq_data(db: Any, seq_id: str, dat: Dict) -> None:

    with db.begin(write=True, buffers=True) as txn:
        txn.put(key=seq_id.encode(), value = pickle.dumps(dat))


def read_seq_data(db: Any, seq_id: str) -> Dict:

    with db.begin(write = False) as txn:
        value = txn.get(seq_id.encode())
        if value is None:
            raise KeyError(f"Sequence '{seq_id}' not found in LMDB")
        enzyme_data = pickle.loads(value)
    return enzyme_data

def process_seq_data(seq_config: SEQ_LMDB_CONFIG, device_str: str) -> None:
    """DEPRECATED: Use DataSet/brenda/compute_prot5_lmdb.py instead.
    This function is kept for compatibility but should not be called directly.
    """
    raise NotImplementedError(
        "process_seq_data is deprecated for ProtT5. "
        "Use compute_prot5_lmdb.py to generate ProtT5 LMDB embeddings."
    )
