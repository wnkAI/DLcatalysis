#!/usr/bin/env python3
"""
Compute ProtT5 per-residue embeddings and store in LMDB.
Output format is identical to ESM2 LMDB (seq_process.py),
so DLCatalysis can swap between the two via config only.

Usage:
    python compute_prot5_lmdb.py \
        --seq_csv  ../DataSet/brenda/final_seq.csv \
        --output   ../DataSet/final_data/seq_prot5.lmdb \
        --model    Rostlab/prot_t5_xl_uniref50 \
        --device   cuda:0 \
        --max_seq_len 1000 \
        --batch_size  8
"""

import argparse
import sys
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from rich.progress import track
from transformers import T5EncoderModel, T5Tokenizer

# Support running from DataSet/ or DataSet/brenda/
_this_dir = Path(__file__).resolve().parent
for _candidate in [_this_dir.parent, _this_dir.parent.parent]:
    _src = _candidate / "src"
    if _src.exists():
        sys.path.insert(0, str(_src))
        break
from util.data_load import SequenceData, padding_seq_embedding


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ProtT5 LMDB embeddings")
    parser.add_argument("--seq_csv", type=str, required=True,
                        help="CSV with columns SEQ_ID, SEQUENCE")
    parser.add_argument("--output", type=str, required=True,
                        help="Output LMDB path")
    parser.add_argument("--model", type=str, default="Rostlab/prot_t5_xl_uniref50",
                        help="ProtT5 model name or local path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_seq_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--map_size", type=int, default=600 * 1024 * 1024 * 1024,
                        help="LMDB map size in bytes (default 600GB)")
    return parser.parse_args()


def load_prot5(model_name: str, device: str):
    """Load ProtT5 encoder in half precision."""
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device).eval()
    return model, tokenizer


def embed_batch(model, tokenizer, sequences, device, max_seq_len):
    """Compute per-residue embeddings for a batch of sequences."""
    # ProtT5 requires spaces between amino acids
    spaced = [" ".join(list(seq[:max_seq_len])) for seq in sequences]
    # Replace non-standard AAs
    spaced = [s.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X") for s in spaced]

    encoding = tokenizer.batch_encode_plus(
        spaced, add_special_tokens=True, padding="longest",
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    embeddings = output.last_hidden_state  # (B, L+1, 1024)
    return embeddings, attention_mask


def main():
    args = parse_args()

    print(f"Loading ProtT5 model: {args.model}")
    model, tokenizer = load_prot5(args.model, args.device)
    emb_dim = model.config.d_model  # 1024 for ProtT5-XL
    print(f"Embedding dim: {emb_dim}")

    df = pd.read_csv(args.seq_csv)
    assert list(df.columns[:2]) == ["SEQ_ID", "SEQUENCE"], \
        "CSV must have columns: SEQ_ID, SEQUENCE"
    print(f"Total sequences: {len(df)}")

    env = lmdb.open(
        args.output, map_size=args.map_size,
        create=True, subdir=False, readonly=False
    )

    n_done = 0
    batch_ids, batch_seqs = [], []

    for i in track(range(len(df)), description="Computing ProtT5 embeddings"):
        seq_id = str(df.loc[i, "SEQ_ID"])
        sequence = str(df.loc[i, "SEQUENCE"])
        batch_ids.append(seq_id)
        batch_seqs.append(sequence)

        if len(batch_ids) >= args.batch_size or i == len(df) - 1:
            try:
                embeddings, masks = embed_batch(
                    model, tokenizer, batch_seqs, args.device, args.max_seq_len
                )

                for j, sid in enumerate(batch_ids):
                    seq = batch_seqs[j]
                    seq_len = min(len(seq), args.max_seq_len)
                    # Slice off padding and EOS token
                    emb = embeddings[j, :seq_len, :].float().cpu().numpy()

                    dat = {
                        "embedding": emb,          # (seq_len, 1024)
                        "sequence": seq[:args.max_seq_len]
                    }
                    dat = padding_seq_embedding(dat, args.max_seq_len)
                    seq_data = SequenceData.from_dict(dat)

                    with env.begin(write=True) as txn:
                        txn.put(sid.encode(), pickle.dumps(seq_data))

                    n_done += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch starting {batch_ids[0]}, trying one by one...")
                    torch.cuda.empty_cache()
                    for j, sid in enumerate(batch_ids):
                        try:
                            embs, _ = embed_batch(
                                model, tokenizer, [batch_seqs[j]],
                                args.device, args.max_seq_len
                            )
                            seq = batch_seqs[j]
                            seq_len = min(len(seq), args.max_seq_len)
                            emb = embs[0, :seq_len, :].float().cpu().numpy()
                            dat = {"embedding": emb, "sequence": seq[:args.max_seq_len]}
                            dat = padding_seq_embedding(dat, args.max_seq_len)
                            seq_data = SequenceData.from_dict(dat)
                            with env.begin(write=True) as txn:
                                txn.put(sid.encode(), pickle.dumps(seq_data))
                            n_done += 1
                        except RuntimeError:
                            print(f"Skipped {sid} (too long or OOM)")
                            torch.cuda.empty_cache()
                else:
                    raise e

            batch_ids, batch_seqs = [], []

    env.close()
    print(f"\nDone! {n_done}/{len(df)} sequences saved to {args.output}")
    print(f"Embedding dim: {emb_dim} — set seq_hidden_dim accordingly in config")


if __name__ == "__main__":
    main()
