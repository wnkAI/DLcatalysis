#!/usr/bin/env python
"""
DLCatalysis unified data preprocessing pipeline.

Steps:
  1. Parse BRENDA JSON → final_data.csv, final_seq.csv, final_smi.csv
  2. Compute Morgan fingerprints → morgan_fingerprint.npy
  3. Generate GROVER substrate CSV + run_grover.sh
  4. CD-HIT 40% k-fold split (10 folds) → fold_0.csv ... fold_9.csv
  5. Print reminder to run ProtT5/MolT5/GROVER precomputation separately

Usage:
    cd DataSet/brenda
    python process.py [--identity 0.40] [--n_folds 10] [--seed 42]
"""

import sys
import os
import argparse
import subprocess
import tempfile
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rich import print as rp
from rich.progress import track

from util.brenda_tools import process_brenda_json


# ── CD-HIT helpers ────────────────────────────────────────────────────────────

def _write_fasta(seq_df: pd.DataFrame, fasta_fp: str) -> None:
    with open(fasta_fp, 'w') as f:
        for _, row in seq_df.iterrows():
            f.write(f">{row['SEQ_ID']}\n{row['SEQUENCE']}\n")


def _parse_cdhit_clstr(clstr_fp: str) -> dict:
    seq_to_cluster = {}
    cluster_id = -1
    with open(clstr_fp) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                cluster_id = int(line.split()[-1])
            elif '>' in line:
                seq_id = line.split('>')[1].split('...')[0]
                seq_to_cluster[seq_id] = cluster_id
    return seq_to_cluster


def _ec1_label(ec_str) -> str:
    try:
        s = str(ec_str).strip()
        if s in ('nan', '-', '', 'None'):
            return '0'
        return s.split('.')[0]
    except Exception:
        return '0'


def _run_cdhit(seq_df, identity, tmpdir):
    """Run CD-HIT and return {seq_id: cluster_id} mapping."""
    fasta_fp = os.path.join(tmpdir, "seqs.fasta")
    clstr_out = os.path.join(tmpdir, "cdhit_out")

    _write_fasta(seq_df, fasta_fp)

    if identity >= 0.7:
        n_word = 5
    elif identity >= 0.6:
        n_word = 4
    elif identity >= 0.5:
        n_word = 3
    else:
        n_word = 2

    cmd = [
        "cd-hit", "-i", fasta_fp, "-o", clstr_out,
        "-c", str(identity), "-n", str(n_word),
        "-T", "8", "-M", "16000", "-d", "0", "-aS", "0.8",
    ]
    rp(f"[cyan]Running CD-HIT at {identity*100:.0f}% identity...[/cyan]")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "cd-hit not found. Install with: conda install -c bioconda cd-hit\n"
            "Or: apt-get install -y cd-hit"
        )
    if result.returncode != 0:
        raise RuntimeError(f"CD-HIT failed:\n{result.stderr}")

    return _parse_cdhit_clstr(clstr_out + ".clstr")


def _cluster_substrates(smi_df, threshold=0.40):
    """Cluster substrates by Tanimoto similarity on Morgan fingerprints."""
    from rdkit import Chem
    from rdkit.ML.Cluster import Butina

    rp(f"[cyan]Clustering substrates at Tanimoto threshold {threshold:.0%}...[/cyan]")
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    fps, valid_indices = [], []
    smi_ids = smi_df['SMI_ID'].tolist()
    for i, smi in enumerate(smi_df['SMILES'].tolist()):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps.append(morgan_gen.GetFingerprint(mol))
            valid_indices.append(i)

    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, 1.0 - threshold, isDistData=True)

    smi_to_cluster = {}
    next_cl = 0
    for cluster in clusters:
        for idx in cluster:
            smi_to_cluster[smi_ids[valid_indices[idx]]] = next_cl
        next_cl += 1
    for i in range(len(smi_ids)):
        if smi_ids[i] not in smi_to_cluster:
            smi_to_cluster[smi_ids[i]] = next_cl
            next_cl += 1

    rp(f"[green]Substrate clusters: {next_cl} (from {len(smi_ids)} substrates)[/green]")
    return smi_to_cluster


# ── K-fold split ──────────────────────────────────────────────────────────────

def cdhit_kfold(final_data, seq_df, identity=0.40, n_folds=10, seed=42,
                out_dir="../final_data", smi_df=None, smi_threshold=0.40):
    """
    CD-HIT cluster-based k-fold split with EC1-stratified greedy assignment.
    Guarantees no enzyme sequence leakage across folds.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        seq_to_cluster = _run_cdhit(seq_df, identity, tmpdir)

    data = final_data.copy()
    data['seq_cluster'] = data['SEQ_ID'].map(seq_to_cluster)
    unmapped = data['seq_cluster'].isna().sum()
    if unmapped > 0:
        rp(f"[yellow]Warning: {unmapped} rows unmapped by CD-HIT, dropping.[/yellow]")
        data = data.dropna(subset=['seq_cluster'])
    data['seq_cluster'] = data['seq_cluster'].astype(int)

    # Substrate clustering (for balance refinement)
    if smi_df is not None and len(smi_df) > 0:
        smi_to_cluster = _cluster_substrates(smi_df, threshold=smi_threshold)
        data['smi_cluster'] = data['SMI_ID'].map(smi_to_cluster).fillna(-1).astype(int)
    else:
        data['smi_cluster'] = 0

    # EC1-stratified greedy balanced assignment
    data['_ec1'] = data['EC_NUMBER'].apply(_ec1_label)
    cluster_ec = (
        data.groupby('seq_cluster')['_ec1']
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={'_ec1': 'ec1_major'})
    )
    cluster_sizes = data.groupby('seq_cluster').size().to_dict()
    cluster_to_fold = {}
    fold_sizes = [0] * n_folds

    for ec1, group in cluster_ec.groupby('ec1_major'):
        clids = sorted(group['seq_cluster'].values,
                       key=lambda c: cluster_sizes.get(c, 0), reverse=True)
        for cid in clids:
            min_fold = min(range(n_folds), key=lambda f: fold_sizes[f])
            cluster_to_fold[cid] = min_fold
            fold_sizes[min_fold] += cluster_sizes.get(cid, 0)

    data['fold'] = data['seq_cluster'].map(cluster_to_fold)

    # Save folds
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    drop_cols = ['seq_cluster', 'smi_cluster', '_ec1', 'fold']

    fold_sizes_list = []
    for k in range(n_folds):
        fold_df = data[data['fold'] == k].drop(columns=drop_cols).reset_index(drop=True)
        fold_df.to_csv(out_path / f"fold_{k}.csv", index=False)
        fold_sizes_list.append(len(fold_df))

    all_df = data.drop(columns=['seq_cluster', 'smi_cluster', '_ec1']).reset_index(drop=True)
    all_df.to_csv(out_path / "all_folds.csv", index=False)

    rp(f"[green]k-fold: {n_folds} folds, identity={identity*100:.0f}%, clusters={len(cluster_ec)}[/green]")
    rp(f"[green]Fold sizes: {fold_sizes_list}[/green]")

    # EC1 distribution per fold
    rp("[cyan]EC1 distribution per fold:[/cyan]")
    for k in range(n_folds):
        dist = data[data['fold'] == k]['_ec1'].value_counts(normalize=True).sort_index()
        rp(f"  fold {k}: { {kk: f'{v:.1%}' for kk, v in dist.items()} }")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DLCatalysis data preprocessing pipeline")
    parser.add_argument("--json", type=str, default="./kkm_data.json", help="BRENDA JSON file")
    parser.add_argument("--identity", type=float, default=0.40, help="CD-HIT identity threshold")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device for embedding")
    parser.add_argument("--prot5_batch", type=int, default=8, help="ProtT5 batch size")
    parser.add_argument("--molt5_batch", type=int, default=32, help="MolT5 batch size")
    parser.add_argument("--prot5_model", type=str,
                        default="Rostlab/prot_t5_xl_uniref50",
                        help="ProtT5 HuggingFace id or local path")
    parser.add_argument("--molt5_model", type=str,
                        default="laituan245/molt5-base-smiles2caption",
                        help="MolT5 HuggingFace id or local path; base=768D matches training config")
    parser.add_argument("--max_seq_len", type=int, default=1000)
    parser.add_argument("--skip_prot5", action="store_true",
                        help="Skip ProtT5 embedding (reuse existing LMDB)")
    parser.add_argument("--skip_molt5", action="store_true",
                        help="Skip MolT5 embedding (reuse existing npy)")
    args = parser.parse_args()

    out_dir = "../final_data"
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Parse BRENDA JSON ────────────────────────────────────────
    rp("[bold cyan]Step 1: Parsing BRENDA JSON...[/bold cyan]")
    data_fp = "./final_data.csv"
    seq_fp = "./final_seq.csv"
    smi_fp = "./final_smi.csv"
    process_brenda_json(args.json, data_fp, seq_fp, smi_fp)

    final_data = pd.read_csv(data_fp)
    seq_df = pd.read_csv(seq_fp)
    smi_df = pd.read_csv(smi_fp)
    rp(f"[green]  Data: {len(final_data)} samples, {len(seq_df)} sequences, {len(smi_df)} substrates[/green]")

    # ── Step 2: Morgan fingerprints ──────────────────────────────────────
    rp("[bold cyan]Step 2: Computing Morgan fingerprints...[/bold cyan]")
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    smi_clean = smi_df.dropna(subset=['SMILES'])
    morgan_results = []
    for substrate in track(smi_clean['SMILES'], description="Morgan FP..."):
        mol = AllChem.MolFromSmiles(substrate)
        if mol is None:
            rp(f"[yellow]Warning: invalid SMILES '{substrate}', using zero FP[/yellow]")
            morgan_results.append(np.zeros((1024,), dtype=np.int8))
            continue
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((1024,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        morgan_results.append(arr)
    np.save(f"{out_dir}/morgan_fingerprint.npy", np.array(morgan_results))
    rp(f"[green]  Saved morgan_fingerprint.npy ({len(morgan_results)} substrates)[/green]")

    # ── Step 3: GROVER (run separately via `bash run_grover.sh`) ─────────
    # GROVER substrate CSV and fingerprints are generated by run_grover.sh,
    # which is self-contained (reads final_smi.csv, writes grover_substrates.csv,
    # then calls grover_software). No internet required.

    # ── Step 4: ProtT5 embeddings ──────────────────────────────────────
    prot5_lmdb = f"{out_dir}/seq_prot5.lmdb"
    lmdb_complete = False
    if os.path.exists(prot5_lmdb):
        try:
            import lmdb as _lmdb_check
            _env = _lmdb_check.open(prot5_lmdb, readonly=True, lock=False, subdir=False)
            with _env.begin() as _txn:
                _n = _txn.stat()["entries"]
            _env.close()
            lmdb_complete = (_n >= len(seq_df))
            rp(f"  Existing LMDB has {_n} entries, seq_df has {len(seq_df)} sequences "
               f"({'complete' if lmdb_complete else 'INCOMPLETE'})")
        except Exception as e:
            rp(f"[yellow]  Could not stat LMDB ({e}); will recompute[/yellow]")

    if args.skip_prot5:
        rp(f"[green]Step 4: ProtT5 LMDB skipped by --skip_prot5 ({prot5_lmdb})[/green]")
    elif lmdb_complete:
        rp(f"[green]Step 4: ProtT5 LMDB up-to-date, skipping ({prot5_lmdb})[/green]")
    else:
        if os.path.exists(prot5_lmdb):
            rp(f"[yellow]  Removing stale LMDB and recomputing[/yellow]")
            os.remove(prot5_lmdb)
            _lock = prot5_lmdb + "-lock"
            if os.path.exists(_lock):
                os.remove(_lock)
        rp("[bold cyan]Step 4: Computing ProtT5 embeddings...[/bold cyan]")
        import pickle
        import lmdb as lmdb_lib
        import torch
        from transformers import T5EncoderModel, T5Tokenizer

        rp(f"  Loading ProtT5 from {args.prot5_model}...")
        prot5_tokenizer = T5Tokenizer.from_pretrained(args.prot5_model, do_lower_case=False)
        prot5_model = T5EncoderModel.from_pretrained(args.prot5_model, torch_dtype=torch.float16)
        prot5_model = prot5_model.to(args.device).eval()

        env = lmdb_lib.open(prot5_lmdb, map_size=600 * (1024 ** 3), create=True, subdir=False)

        from util.data_load import SequenceData, padding_seq_embedding

        n_done = 0
        batch_ids, batch_seqs = [], []
        for i in track(range(len(seq_df)), description="ProtT5 embedding..."):
            sid = str(seq_df.loc[i, "SEQ_ID"])
            sequence = str(seq_df.loc[i, "SEQUENCE"])
            batch_ids.append(sid)
            batch_seqs.append(sequence)

            if len(batch_ids) >= args.prot5_batch or i == len(seq_df) - 1:
                try:
                    spaced = [" ".join(list(s[:args.max_seq_len])).replace("U","X").replace("Z","X").replace("O","X").replace("B","X") for s in batch_seqs]
                    enc = prot5_tokenizer(spaced, add_special_tokens=True, padding="longest", return_tensors="pt")
                    input_ids = enc["input_ids"].to(args.device)
                    attn_mask = enc["attention_mask"].to(args.device)
                    with torch.no_grad(), torch.amp.autocast("cuda"):
                        output = prot5_model(input_ids=input_ids, attention_mask=attn_mask)
                    embeddings = output.last_hidden_state

                    for j, sid_j in enumerate(batch_ids):
                        seq_len = min(len(batch_seqs[j]), args.max_seq_len)
                        emb = embeddings[j, :seq_len, :].float().cpu().numpy()
                        dat = {"embedding": emb, "sequence": batch_seqs[j][:args.max_seq_len]}
                        dat = padding_seq_embedding(dat, args.max_seq_len)
                        seq_data = SequenceData.from_dict(dat)
                        with env.begin(write=True) as txn:
                            txn.put(sid_j.encode(), pickle.dumps(seq_data))
                        n_done += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        rp(f"[yellow]OOM at batch {batch_ids[0]}, trying one by one...[/yellow]")
                        torch.cuda.empty_cache()
                        for j, sid_j in enumerate(batch_ids):
                            try:
                                sp = " ".join(list(batch_seqs[j][:args.max_seq_len])).replace("U","X").replace("Z","X").replace("O","X").replace("B","X")
                                enc = prot5_tokenizer([sp], add_special_tokens=True, padding="longest", return_tensors="pt")
                                with torch.no_grad(), torch.amp.autocast("cuda"):
                                    out = prot5_model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))
                                seq_len = min(len(batch_seqs[j]), args.max_seq_len)
                                emb = out.last_hidden_state[0, :seq_len, :].float().cpu().numpy()
                                dat = {"embedding": emb, "sequence": batch_seqs[j][:args.max_seq_len]}
                                dat = padding_seq_embedding(dat, args.max_seq_len)
                                seq_data = SequenceData.from_dict(dat)
                                with env.begin(write=True) as txn:
                                    txn.put(sid_j.encode(), pickle.dumps(seq_data))
                                n_done += 1
                            except RuntimeError:
                                rp(f"[yellow]Skipped {sid_j} (OOM)[/yellow]")
                                torch.cuda.empty_cache()
                    else:
                        raise
                batch_ids, batch_seqs = [], []

        env.close()
        del prot5_model, prot5_tokenizer
        torch.cuda.empty_cache()
        rp(f"[green]  ProtT5: {n_done}/{len(seq_df)} sequences → {prot5_lmdb}[/green]")

    # ── Step 5: MolT5 embeddings ─────────────────────────────────────────
    molt5_npy = f"{out_dir}/molt5_embedding.npy"
    if args.skip_molt5:
        rp(f"[green]Step 5: MolT5 embeddings skipped[/green]")
    else:
        rp("[bold cyan]Step 5: Computing MolT5 embeddings...[/bold cyan]")
        import torch
        from transformers import T5EncoderModel as T5Enc, T5Tokenizer as T5Tok

        smi_sorted = smi_df.copy()
        smi_sorted["smi_idx"] = smi_sorted["SMI_ID"].str.replace("SMI", "").astype(int)
        smi_sorted = smi_sorted.sort_values("smi_idx").reset_index(drop=True)
        smiles_list = smi_sorted["SMILES"].tolist()

        rp(f"  Loading MolT5 from {args.molt5_model}...")
        molt5_tok = T5Tok.from_pretrained(args.molt5_model, model_max_length=512)
        molt5_mdl = T5Enc.from_pretrained(args.molt5_model).to(args.device).eval()

        all_embs = []
        for i in track(range(0, len(smiles_list), args.molt5_batch), description="MolT5 encoding..."):
            batch = smiles_list[i:i + args.molt5_batch]
            enc = molt5_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                out = molt5_mdl(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))
            hidden = out.last_hidden_state
            for j in range(hidden.shape[0]):
                mask = enc["attention_mask"][j].bool()
                real = hidden[j][mask][:-1]  # exclude EOS
                if real.shape[0] == 0:
                    real = hidden[j][mask]
                all_embs.append(real.mean(dim=0).cpu().float().numpy())

        molt5_arr = np.stack(all_embs, axis=0).astype(np.float32)
        np.save(molt5_npy, molt5_arr)
        del molt5_mdl, molt5_tok
        torch.cuda.empty_cache()
        rp(f"[green]  MolT5: {molt5_arr.shape} → {molt5_npy}[/green]")

    # ── Step 6: CD-HIT k-fold split ──────────────────────────────────────
    rp(f"[bold cyan]Step 6: CD-HIT {args.identity*100:.0f}% {args.n_folds}-fold split...[/bold cyan]")
    cdhit_kfold(
        final_data, seq_df,
        identity=args.identity,
        n_folds=args.n_folds,
        seed=args.seed,
        out_dir=out_dir,
        smi_df=smi_df,
    )

    # ── Done ─────────────────────────────────────────────────────────────
    rp("")
    rp("[bold green]═══ Pipeline complete! ═══[/bold green]")
    rp(f"[green]  Output: {out_dir}/[/green]")
    rp("[yellow]  GROVER fingerprints still need separate run: bash run_grover.sh[/yellow]")
    rp("")


if __name__ == "__main__":
    main()
