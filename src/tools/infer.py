#!/usr/bin/env python
"""
DLCatalysis 3.0 inference script — 5 features computed on-the-fly.

Given sequence + SMILES + EC number, computes all features in realtime
(identical to training pipeline). No precomputed LMDB/npy needed.

Features computed:
    1. ProtT5  — protein sequence encoder  (1000, 1024)
    2. Morgan  — molecular fingerprint     (1024,)
    3. MolT5   — molecular language model  (768,)
    4. GROVER  — graph neural network      (4885,)  = 4*1200 + 85 fgtasklabel
    5. EC      — enzyme commission number  4-level embedding

Input XLSX:
    seq_id | sequence | smiles | ec_number

Usage:
    python infer.py \\
        --config ../../config/config.yml \\
        --ckpt_dir ../../train/best_ckpts \\
        --input ../../BF_input.xlsx \\
        --output ../../BF_predictions.xlsx
"""
import sys
import os
import argparse
import gc
import glob
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model.enzsub import EnzSub
from util.tools import init_config
from util.data_module import SeqMolComplexData, _parse_ec_number


# ═══════════════════════════════════════════════════════════════════════
# 1. ProtT5 — protein sequence embedding
# ═══════════════════════════════════════════════════════════════════════

def compute_prott5_embeddings(unique_seqs, config, device='cpu'):
    """Load ProtT5, compute embeddings for unique sequences, then free model.

    Uses raw output.last_hidden_state (NO layer fusion), identical to
    DataSet/brenda/compute_prot5_lmdb.py which built the training LMDB.
    """
    from transformers import T5EncoderModel, T5Tokenizer

    max_len = config['data']['seq_lmdb']['max_seq_len']
    model_path = config['model'].get('prot5_model_path',
                                     'Rostlab/prot_t5_xl_uniref50')
    if not os.path.exists(model_path):
        model_path = 'Rostlab/prot_t5_xl_uniref50'
    print(f"  Loading ProtT5 from {model_path} ...")

    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32)
    model = model.to(device).eval()

    def _sanitize(aa):
        return "X" if aa in ("U", "Z", "O", "B") else aa

    cache = {}
    for i, seq in enumerate(unique_seqs):
        seq_trunc = seq[:max_len]
        seq_len = len(seq_trunc)

        spaced = " ".join(_sanitize(aa) for aa in seq_trunc)
        enc = tokenizer(spaced, add_special_tokens=True, padding="longest",
                        return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids,
                           attention_mask=attention_mask)
        emb = output.last_hidden_state[0, :seq_len, :].cpu().float()

        if emb.shape[0] < max_len:
            pad = torch.zeros(max_len - emb.shape[0], emb.shape[1])
            emb = torch.cat([emb, pad], dim=0)
        mask = np.ones((1, max_len), dtype=bool)
        mask[0, :seq_len] = False
        cache[seq] = {
            'embedding': emb,
            'seq_padding_mask': torch.tensor(mask, dtype=torch.bool),
            'sequence': seq_trunc,
        }
        if (i + 1) % 10 == 0 or i == len(unique_seqs) - 1:
            print(f"  ProtT5: {i + 1}/{len(unique_seqs)}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


# ═══════════════════════════════════════════════════════════════════════
# 2. Morgan — molecular fingerprint
# ═══════════════════════════════════════════════════════════════════════

def compute_morgan(smiles, radius=2, n_bits=1024):
    """Compute Morgan fingerprint for a single SMILES via RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  Warning: invalid SMILES for Morgan: {smiles[:50]}")
        return torch.zeros(n_bits, dtype=torch.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor(list(fp), dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════
# 3. MolT5 — molecular language model embedding
# ═══════════════════════════════════════════════════════════════════════

def compute_molt5_embeddings(unique_smiles, config, device='cpu'):
    """Load MolT5, compute embeddings for unique SMILES, then free model.

    Pooling: mean over real tokens EXCLUDING padding AND EOS,
    identical to DataSet/brenda/compute_molt5.py (training precomputation).
    """
    from transformers import T5EncoderModel, T5Tokenizer

    model_path = config['model'].get('molt5_model_path', 'laituan245/molt5-base')
    if not os.path.exists(model_path):
        model_path = 'laituan245/molt5-base'
    print(f"  Loading MolT5 from {model_path} ...")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    cache = {}
    for i, smi in enumerate(unique_smiles):
        enc = tokenizer(smi, return_tensors='pt', padding=True,
                        truncation=True, max_length=512)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[0]
        mask = attention_mask[0].bool()
        real_tokens = hidden[mask][:-1]
        if real_tokens.shape[0] == 0:
            real_tokens = hidden[mask]
        emb = real_tokens.mean(dim=0).cpu().float()
        cache[smi] = emb
        if (i + 1) % 50 == 0 or i == len(unique_smiles) - 1:
            print(f"  MolT5: {i + 1}/{len(unique_smiles)}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


# ═══════════════════════════════════════════════════════════════════════
# 4. GROVER — graph-level molecular fingerprint + fgtasklabel features
# ═══════════════════════════════════════════════════════════════════════

def compute_grover_embeddings(unique_smiles, grover_ckpt, device='cpu',
                              grover_dir=None):
    """Load GROVER, compute fingerprints for unique SMILES, then free model.

    Replicates the exact pipeline from run_grover.sh:
        1. Compute fgtasklabel features (85-dim RDKit functional group binary)
        2. Build molecular graph (MolGraph -> BatchMolGraph)
        3. Forward through GroverFpGeneration (fingerprint_source='both')
        4. Output = concat(4 x readout, fgtasklabel) = 4*hidden_size + 85
    """
    if grover_dir is None or not os.path.exists(grover_dir):
        grover_dir = os.path.join(
            PROJECT_ROOT.parent, 'other_softwares', 'grover_software')
    if not os.path.exists(grover_dir):
        raise FileNotFoundError(
            f"GROVER software not found at {grover_dir}. "
            f"Set model.grover_dir in the yml or pass --grover_dir.")
    if grover_dir not in sys.path:
        sys.path.insert(0, grover_dir)

    from grover.util.utils import load_checkpoint
    from grover.data.molgraph import MolGraph, BatchMolGraph
    from grover.data.task_labels import rdkit_functional_group_label_features_generator

    print(f"  Loading GROVER from {grover_ckpt} ...")
    use_cuda = (device != 'cpu' and torch.cuda.is_available())

    fp_args = Namespace(
        parser_name='fingerprint',
        fingerprint_source='both',
        cuda=use_cuda,
        no_cache=True,
        bond_drop_rate=0,
    )
    model = load_checkpoint(grover_ckpt, current_args=fp_args, cuda=use_cuda)
    model.eval()
    fp_args.bond_drop_rate = 0

    cache = {}
    batch_size = 32
    smiles_list = list(unique_smiles)

    for start in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[start:start + batch_size]

        mol_graphs = []
        features_batch = []
        valid_mask = []
        for smi in batch_smiles:
            try:
                mol_graphs.append(MolGraph(smi, fp_args))
                features_batch.append(
                    rdkit_functional_group_label_features_generator(smi))
                valid_mask.append(True)
            except Exception as e:
                print(f"  Warning: GROVER graph build failed for '{smi[:40]}': {e}")
                valid_mask.append(False)

        if mol_graphs:
            batch_mol_graph = BatchMolGraph(mol_graphs, fp_args)
            batch = batch_mol_graph.get_components()
            with torch.no_grad():
                _, _, fp = model(batch, features_batch)

            valid_idx = 0
            for j, smi in enumerate(batch_smiles):
                if valid_mask[j]:
                    cache[smi] = fp[valid_idx].cpu().float()
                    valid_idx += 1
                else:
                    grover_dim = fp.shape[1] if fp.shape[0] > 0 else 4885
                    cache[smi] = torch.zeros(grover_dim, dtype=torch.float32)
        else:
            for smi in batch_smiles:
                cache[smi] = torch.zeros(4885, dtype=torch.float32)

        done = min(start + batch_size, len(smiles_list))
        if done % 100 == 0 or done == len(smiles_list):
            print(f"  GROVER: {done}/{len(smiles_list)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


# ═══════════════════════════════════════════════════════════════════════
# Assemble SeqMolComplexData (identical to training data_module.py)
# ═══════════════════════════════════════════════════════════════════════

def build_data(sequence, smiles, ec_number, config,
               prott5_cache, morgan_cache, molt5_cache, grover_cache):
    """Assemble features into a SeqMolComplexData, same as training."""

    cfg_emb = config['model']['embedding']

    seq_data = prott5_cache[sequence]

    mol_dict = {}
    if cfg_emb.get('morgan', False):
        mol_dict['morgan'] = morgan_cache[smiles]
    if cfg_emb.get('molt5', False):
        mol_dict['molt5'] = molt5_cache[smiles]
    if cfg_emb.get('grover', False):
        mol_dict['grover_mean'] = grover_cache[smiles]

    data = SeqMolComplexData.from_dict(sequence=seq_data, molecular=mol_dict)

    if config['model'].get('use_ec', False):
        ec_valid = (ec_number is not None
                    and not (isinstance(ec_number, float) and np.isnan(ec_number))
                    and str(ec_number).strip() not in ('', 'nan', 'None'))
        ec_ids = _parse_ec_number(ec_number) if ec_valid else [0, 0, 0, 0]
        data.EC_ids = torch.tensor([ec_ids], dtype=torch.long)

    data.y = torch.tensor([0.0], dtype=torch.float)
    return data


# ═══════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_batch(model, data_list, device, batch_size=4):
    model.eval()
    all_preds = []
    for i in range(0, len(data_list), batch_size):
        batch = Batch.from_data_list(data_list[i:i + batch_size]).to(device)
        y_pred, _ = model(batch)
        all_preds.extend(y_pred.squeeze(-1).cpu().numpy().tolist())
    return np.array(all_preds)


def find_ckpts(ckpt_dir):
    ckpt_paths = []
    for fold in range(10):
        fold_dir = os.path.join(ckpt_dir, f"fold{fold}")
        if not os.path.exists(fold_dir):
            continue
        ckpts = [f for f in glob.glob(
            os.path.join(fold_dir, "**", "*.ckpt"), recursive=True)
                 if "last" not in os.path.basename(f)]
        if len(ckpts) > 1:
            ckpts = [max(ckpts, key=os.path.getmtime)]
        if ckpts:
            ckpt_paths.append(ckpts[0])
    if not ckpt_paths:
        ckpt_paths = sorted([
            f for f in glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
            if "last" not in os.path.basename(f)])
    return ckpt_paths


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='DLCatalysis 3.0 — realtime inference (all features on-the-fly)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Single checkpoint path')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Directory with fold0/ ... fold9/ checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grover_ckpt', type=str, default=None,
                        help='Path to grover_large.pt '
                             '(default: DataSet/checkpoint/grover_large.pt)')
    parser.add_argument('--grover_dir', type=str, default=None,
                        help='Path to GROVER source directory '
                             '(default: other_softwares/grover_software)')

    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--smiles', type=str, default=None)
    parser.add_argument('--ec', type=str, default=None)
    parser.add_argument('--seq_id', type=str, default=None)

    parser.add_argument('--input', type=str, default=None,
                        help='XLSX/CSV with columns: seq_id, sequence, smiles'
                             ' [, ec_number]')
    parser.add_argument('--output', type=str, default='predictions.xlsx')
    args = parser.parse_args()

    if not args.ckpt and not args.ckpt_dir:
        parser.error("Must provide --ckpt or --ckpt_dir")

    config = init_config(args.config)
    device = args.device if torch.cuda.is_available() else 'cpu'
    cfg_emb = config['model']['embedding']

    print("\n[Config Check]")
    print(f"  hidden_dim:      {config['model']['hidden_dim']}")
    print(f"  use_ec:          {config['model'].get('use_ec', False)}")
    print(f"  cross_attention: "
          f"{'disabled' if config['model'].get('cross_attention', {}).get('disable', False) else 'enabled'}")
    for feat in ('morgan', 'molt5', 'grover'):
        print(f"  {feat:10s} {cfg_emb.get(feat, False)}"
              f"  dim={cfg_emb.get(feat + '_dim', '?')}")

    grover_ckpt = args.grover_ckpt
    if grover_ckpt is None:
        grover_ckpt = config['model'].get('grover_ckpt_path')
    if not grover_ckpt or not os.path.exists(grover_ckpt):
        fallback = os.path.join(
            PROJECT_ROOT.parent, 'DataSet', 'checkpoint', 'grover_large.pt')
        if grover_ckpt and not os.path.exists(grover_ckpt):
            print(f"  Warning: grover_ckpt '{grover_ckpt}' not found, "
                  f"falling back to {fallback}")
        grover_ckpt = fallback
    if cfg_emb.get('grover', False) and not os.path.exists(grover_ckpt):
        raise FileNotFoundError(
            f"GROVER enabled but ckpt not found at '{grover_ckpt}'. "
            f"Pass --grover_ckpt or set model.grover_ckpt_path in the yml.")

    grover_dir = args.grover_dir or config['model'].get('grover_dir')
    if not grover_dir or not os.path.exists(grover_dir):
        fallback_dir = os.path.join(
            PROJECT_ROOT.parent, 'other_softwares', 'grover_software')
        if grover_dir and not os.path.exists(grover_dir):
            print(f"  Warning: grover_dir '{grover_dir}' not found, "
                  f"falling back to {fallback_dir}")
        grover_dir = fallback_dir
    if cfg_emb.get('grover', False) and not os.path.exists(grover_dir):
        raise FileNotFoundError(
            f"GROVER enabled but source dir not found at '{grover_dir}'. "
            f"Pass --grover_dir or set model.grover_dir in the yml.")

    if args.ckpt_dir:
        ckpt_paths = find_ckpts(args.ckpt_dir)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoints in {args.ckpt_dir}")
        print(f"Found {len(ckpt_paths)} fold checkpoints")
    else:
        ckpt_paths = [args.ckpt]

    if args.sequence and args.smiles:
        df = pd.DataFrame([{
            'seq_id': args.seq_id or 'query',
            'sequence': args.sequence,
            'smiles': args.smiles,
            'ec_number': args.ec,
        }])
    elif args.input:
        if args.input.endswith('.xlsx'):
            df = pd.read_excel(args.input)
        else:
            df = pd.read_csv(args.input)
        if 'sequence' not in df.columns or 'smiles' not in df.columns:
            raise ValueError("Input must have 'sequence' and 'smiles' columns")
        if 'seq_id' not in df.columns:
            df['seq_id'] = [f"sample_{i}" for i in range(len(df))]
    else:
        parser.print_help()
        return

    if df['sequence'].isna().any() or df['smiles'].isna().any():
        raise ValueError("Input contains missing values in 'sequence' or 'smiles'.")
    df['sequence'] = df['sequence'].astype(str)
    df['smiles'] = df['smiles'].astype(str)
    if (df['sequence'].str.strip() == '').any():
        raise ValueError("Input contains empty 'sequence' values.")
    if (df['smiles'].str.strip() == '').any():
        raise ValueError("Input contains empty 'smiles' values.")

    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    def _standardize_smiles(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi[:60]}")
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        return Chem.MolToSmiles(mol)

    print("\nStandardizing SMILES (desalt + canonicalize) ...")
    df['smiles_raw'] = df['smiles']
    df['smiles'] = df['smiles'].apply(_standardize_smiles)
    n_changed = (df['smiles'] != df['smiles_raw']).sum()
    if n_changed > 0:
        print(f"  {n_changed}/{len(df)} SMILES changed after standardization")

    unique_seqs = list(dict.fromkeys(df['sequence'].tolist()))
    unique_smiles = list(dict.fromkeys(df['smiles'].tolist()))
    print(f"\n{len(df)} samples, {len(unique_seqs)} unique sequences, "
          f"{len(unique_smiles)} unique SMILES")

    print("\n" + "=" * 60)
    print("Phase 1: Computing features (identical to training)")
    print("=" * 60)

    print(f"\n[1/4] ProtT5 ({len(unique_seqs)} sequences) ...")
    prott5_cache = compute_prott5_embeddings(unique_seqs, config, device=device)

    morgan_cache = {}
    if cfg_emb.get('morgan', False):
        morgan_dim = cfg_emb.get('morgan_dim', 1024)
        print(f"\n[2/4] Morgan fingerprint ({len(unique_smiles)} SMILES, "
              f"{morgan_dim}-bit, radius 2) ...")
        for smi in unique_smiles:
            morgan_cache[smi] = compute_morgan(smi, n_bits=morgan_dim)
        print(f"  Morgan: done")
    else:
        print("\n[2/4] Morgan: disabled in config")

    if cfg_emb.get('molt5', False):
        print(f"\n[3/4] MolT5 ({len(unique_smiles)} SMILES) ...")
        molt5_cache = compute_molt5_embeddings(unique_smiles, config,
                                               device=device)
    else:
        molt5_cache = {}
        print("\n[3/4] MolT5: disabled in config")

    if cfg_emb.get('grover', False):
        print(f"\n[4/4] GROVER ({len(unique_smiles)} SMILES) ...")
        grover_cache = compute_grover_embeddings(unique_smiles, grover_ckpt,
                                                 device=device,
                                                 grover_dir=grover_dir)
    else:
        grover_cache = {}
        print("\n[4/4] GROVER: disabled in config")

    sample_seq = unique_seqs[0]
    sample_smi = unique_smiles[0]
    print("\n[Feature Verification]")
    print(f"  ProtT5:  {prott5_cache[sample_seq]['embedding'].shape}")
    if morgan_cache:
        print(f"  Morgan:  {morgan_cache[sample_smi].shape}")
    if molt5_cache:
        print(f"  MolT5:   {molt5_cache[sample_smi].shape}")
    if grover_cache:
        print(f"  GROVER:  {grover_cache[sample_smi].shape}")
    print(f"  EC:      4-level embedding")

    print("\n" + "=" * 60)
    print("Phase 2: Assembling SeqMolComplexData")
    print("=" * 60)

    data_list = []
    for _, row in df.iterrows():
        data_list.append(build_data(
            sequence=str(row['sequence']),
            smiles=str(row['smiles']),
            ec_number=row.get('ec_number'),
            config=config,
            prott5_cache=prott5_cache,
            morgan_cache=morgan_cache,
            molt5_cache=molt5_cache,
            grover_cache=grover_cache,
        ))
    print(f"  {len(data_list)} data objects assembled")

    del prott5_cache, morgan_cache, molt5_cache, grover_cache
    gc.collect()

    print("\n" + "=" * 60)
    print(f"Phase 3: Ensemble prediction ({len(ckpt_paths)} folds)")
    print("=" * 60)

    def _get_nested(d, dotkey):
        for k in dotkey.split('.'):
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return None
        return d

    _test_ckpt = torch.load(ckpt_paths[0], map_location='cpu', weights_only=False)
    if 'hyper_parameters' in _test_ckpt and 'config' in _test_ckpt['hyper_parameters']:
        ckpt_cfg = _test_ckpt['hyper_parameters']['config']
        STRICT_KEYS = [
            'model.hidden_dim', 'model.use_ec',
            'model.embedding.morgan', 'model.embedding.molt5',
            'model.embedding.grover',
            'model.embedding.morgan_dim', 'model.embedding.molt5_dim',
            'model.embedding.grover_dim',
            'model.cross_attention.disable', 'model.cross_attention.n_head',
            'model.cross_attention.n_layers',
            'model.output_header.num_layers', 'model.output_header.hidden_dim',
            'model.seq_module.seq_hidden_dim',
        ]
        mismatches = []
        for key in STRICT_KEYS:
            v_cfg = _get_nested(config, key)
            v_ckpt = _get_nested(ckpt_cfg, key)
            if v_cfg != v_ckpt:
                mismatches.append(f"  {key}: config={v_cfg} vs ckpt={v_ckpt}")
        if mismatches:
            raise ValueError(
                "Config does not match checkpoint!\n" + "\n".join(mismatches))
        print("  Config/checkpoint consistency: OK")
    del _test_ckpt

    all_fold_preds = []
    for j, ckpt_path in enumerate(ckpt_paths):
        print(f"  Fold {j}: {os.path.basename(ckpt_path)}")
        model = EnzSub.load_from_checkpoint(
            ckpt_path, config=config, map_location=device)
        model = model.to(device)
        preds = predict_batch(model, data_list, device,
                              batch_size=args.batch_size)
        all_fold_preds.append(preds)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble_preds = np.mean(all_fold_preds, axis=0)

    df['predicted_log10_kcat_km'] = ensemble_preds
    df['predicted_kcat_km'] = 10 ** ensemble_preds

    if args.output.endswith('.xlsx'):
        df.to_excel(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    print(f"\nSaved {len(df)} predictions to {args.output}")
    cols = ['seq_id', 'predicted_log10_kcat_km', 'predicted_kcat_km']
    print(df[cols].head(10).to_string(index=False))

    if len(df) == 1:
        print(f"\nlog10(kcat/Km): {ensemble_preds[0]:.4f}")
        print(f"kcat/Km: {10 ** ensemble_preds[0]:.4e} M-1 s-1")


if __name__ == '__main__':
    main()
