#!/usr/bin/env python
"""
Precompute MolT5 embeddings for all unique SMILES in the dataset.
Output: numpy array of shape (N_smiles, molt5_dim), indexed by SMI integer index.
        Same layout as morgan_fingerprint.npy → drop-in replacement.

Usage:
    python compute_molt5.py \
        --smi_csv  ../brenda/final_smi.csv \
        --output   ../final_data/molt5_embedding.npy \
        --model    laituan245/molt5-small \
        --batch    32 \
        --gpu      0
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer
from pathlib import Path
from rich.progress import track
from rich import print as rp


def get_molt5_embeddings(smiles_list, model, tokenizer, device, batch_size=32):
    """
    Encode a list of SMILES strings with MolT5.
    Returns numpy array of shape (N, hidden_dim).
    Each embedding = mean of all token hidden states (excluding EOS token).
    """
    all_embeddings = []
    for i in track(range(0, len(smiles_list), batch_size), description="MolT5 encoding..."):
        batch = smiles_list[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (B, seq_len, hidden_dim)
        hidden = outputs.last_hidden_state  # (B, L, D)

        for j in range(hidden.shape[0]):
            mask = attention_mask[j].bool()   # (L,)
            # Exclude EOS (last real token): same as CataPro [:-1]
            real_tokens = hidden[j][mask][:-1]   # (real_len-1, D)
            if real_tokens.shape[0] == 0:
                real_tokens = hidden[j][mask]    # fallback: keep all if only 1 token
            emb = real_tokens.mean(dim=0).cpu().float().numpy()
            all_embeddings.append(emb)

    return np.stack(all_embeddings, axis=0)   # (N, D)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smi_csv", type=str,
                        default="../brenda/final_smi.csv")
    parser.add_argument("--output",  type=str,
                        default="../final_data/molt5_embedding.npy")
    parser.add_argument("--model",   type=str,
                        default="laituan245/molt5-small",
                        help="HuggingFace model id: molt5-small(512D) / molt5-base(768D) / molt5-large(1024D)")
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--gpu",     type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    rp(f"[cyan]Device: {device}[/cyan]")

    # ── Load SMILES ──────────────────────────────────────────────────────
    df = pd.read_csv(args.smi_csv)
    # Sort by integer index to ensure numpy row = SMI index
    df["smi_idx"] = df["SMI_ID"].str.replace("SMI", "").astype(int)
    df = df.sort_values("smi_idx").reset_index(drop=True)
    smiles_list = df["SMILES"].tolist()
    rp(f"[green]Loaded {len(smiles_list)} SMILES[/green]")

    # ── Load MolT5 ───────────────────────────────────────────────────────
    rp(f"[cyan]Loading MolT5 model: {args.model}[/cyan]")
    tokenizer = T5Tokenizer.from_pretrained(args.model, model_max_length=512)
    model     = T5EncoderModel.from_pretrained(args.model).to(device).eval()
    rp(f"[green]Model loaded[/green]")

    # ── Compute embeddings ───────────────────────────────────────────────
    embeddings = get_molt5_embeddings(smiles_list, model, tokenizer, device, args.batch)
    rp(f"[green]Embeddings shape: {embeddings.shape}[/green]")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(Path(args.output).parent, exist_ok=True)
    np.save(args.output, embeddings.astype(np.float32))
    rp(f"[bold green]Saved → {args.output}[/bold green]")

    # Sanity check
    loaded = np.load(args.output)
    rp(f"[green]Verified: {loaded.shape}, dtype={loaded.dtype}[/green]")


if __name__ == "__main__":
    main()
