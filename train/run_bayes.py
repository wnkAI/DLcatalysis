#!/usr/bin/env python
"""
run_bayes.py — Bayesian hyperparameter optimization using Optuna (TPE sampler).

Strategy:
  Each trial trains on one or more "proxy" folds (default: all 10) and returns
  the mean test PCC as the objective.
  After n_trials, the best hyperparameter set is saved by the orchestrator.

Usage (from train/ directory):
  python run_bayes.py --n_trials 30 --gpu 0
  python run_bayes.py --n_trials 30 --gpu 0 --proxy_folds 0 1 2   # 3-fold proxy
"""

import argparse
import os
import sys

# ── Set CUDA_VISIBLE_DEVICES BEFORE any CUDA/torch import ─────────────────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=0)
_gpu_arg, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_arg.gpu)
# ─────────────────────────────────────────────────────────────────────────────

import copy
import random

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = "../src"
sys.path.append(PROJECT_ROOT)
from tools.train_model import train
from util.tools import init_config

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
FOLD_DIR = os.environ.get("FOLD_DIR", "../DataSet/final_data")
CONFIG_BASE = "../config/config_best_stage1.yml"
OUT_DIR = os.environ.get("BAYES_OUT_DIR", "./bayes_results")
N_FOLDS = 10


# ── Proxy CSV builder ─────────────────────────────────────────────────────────

def build_proxy_csvs(test_fold: int, tmp_dir: str) -> tuple:
    """train = remaining 9 folds; eval = held-out fold_{test_fold}."""
    train_folds = [i for i in range(N_FOLDS) if i != test_fold]
    eval_csv  = f"{FOLD_DIR}/fold_{test_fold}.csv"
    train_csv = os.path.join(tmp_dir, f"proxy_train_{test_fold}.csv")
    dfs = [pd.read_csv(f"{FOLD_DIR}/fold_{i}.csv") for i in train_folds]
    pd.concat(dfs, ignore_index=True).to_csv(train_csv, index=False)
    return train_csv, eval_csv


# ── Hyperparameter space ──────────────────────────────────────────────────────

def apply_params(params: dict, base_config: dict) -> dict:
    """Apply a flat params dict to a deep-copied base config."""
    cfg = copy.deepcopy(base_config)

    cfg["train"]["optimizer"]["lr"]           = params["lr"]
    cfg["train"]["optimizer"]["weight_decay"] = params["weight_decay"]
    cfg["train"]["optimizer"]["warm_epoch"]   = params["warm_epoch"]

    cfg["model"]["dropout"]          = params["dropout"]
    cfg["model"]["modality_dropout"] = params["modality_dropout"]
    cfg["model"]["label_noise"]      = params["label_noise"]

    cfg["model"]["hidden_dim"]                       = params["hidden_dim"]
    cfg["model"]["seq_module"]["seq_hidden_dim"]     = params["seq_hidden_dim"]
    cfg["model"]["cross_attention"]["n_head"]        = params["n_head"]
    cfg["model"]["output_header"]["num_layers"]      = params["header_layers"]
    cfg["model"]["output_header"]["hidden_dim"]      = params["header_hidden_dim"]

    cfg["data"]["target_mode"] = "raw"
    cfg["data"]["use_qt"] = False

    cfg["train"]["batch_size"] = params["batch_size"]

    cfg["model"]["gate"]                = params.get("gate", True)
    cfg["model"]["embedding"]["grover"] = params.get("use_grover", True)
    cfg["model"]["embedding"]["molt5"]  = params.get("use_molt5", True)
    cfg["model"]["embedding"]["morgan"] = params.get("use_morgan", True)
    cfg["model"]["pcc_loss_weight"]     = params.get("pcc_loss_weight", 0.2)

    cfg["model"]["loss_type"]                  = params.get("loss_type", "logcosh")
    cfg["model"]["skip_attended"]              = params.get("skip_attended", False)
    cfg["model"]["cross_attention"]["disable"] = params.get("cross_attn_disable", False)
    cfg["model"]["use_ec"]                     = params.get("use_ec", True)

    cfg["model"]["rank_loss_weight"]           = params.get("rank_loss_weight", 0.1)
    cfg["model"]["rank_loss_substrate_weight"] = params.get("rank_loss_substrate_weight", 0.1)
    cfg["model"]["rank_margin_enzyme"]         = params.get("rank_margin_enzyme", 0.1)
    cfg["model"]["rank_margin_substrate"]      = params.get("rank_margin_substrate", 0.1)

    return cfg


def suggest_params(trial: optuna.Trial) -> dict:
    """Unified hyperparameter search (9 dims, ranges tightened from prior HPO data)."""
    return {
        # Optim (2)
        "lr":               trial.suggest_float("lr",            5e-5, 2e-4, log=True),
        "weight_decay":     trial.suggest_float("weight_decay",  0.01, 0.10, log=True),
        # Regularisation (2) — discrete grid 0.025
        "dropout":          trial.suggest_float("dropout",       0.25, 0.45, step=0.025),
        "modality_dropout": trial.suggest_float("modality_dropout", 0.05, 0.20, step=0.025),
        # Loss (1) — discrete grid 0.025
        "pcc_loss_weight":  trial.suggest_float("pcc_loss_weight",     0.15, 0.45, step=0.025),
        # Ranking (3) — discrete grid 0.025
        "rank_loss_weight":           trial.suggest_float("rank_loss_weight",    0.00, 0.30, step=0.025),
        "rank_loss_substrate_weight": trial.suggest_float("rank_loss_substrate_weight", 0.00, 0.30, step=0.025),
        "rank_margin_enzyme":         trial.suggest_float("rank_margin_enzyme",  0.05, 0.30, step=0.025),
        # Fixed (converged in prior HPO)
        "loss_type":         "logcosh",
        "hidden_dim":        512,
        "batch_size":        16,
        "seq_hidden_dim":    512,
        "n_head":            4,
        "header_hidden_dim": 256,
        "header_layers":     2,
        "rank_margin_substrate": 0.1,
        # Fixed scheduler / extras
        "warm_epoch":        5,
        "label_noise":       0.0,
        "use_morgan":        True,
        "use_grover":        True,
        "use_molt5":         True,
        "cross_attn_disable": False,
        "use_ec":            True,
        "skip_attended":     False,
        "gate":              True,
    }


# ── Optuna objective ──────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, base_config: dict,
              proxy_folds: list, tmp_dir: str) -> float:
    params = suggest_params(trial)
    pccs, r2s, mses, maes, rmsds = [], [], [], [], []
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    for step, test_fold in enumerate(proxy_folds):
        train_csv, eval_csv = build_proxy_csvs(test_fold, tmp_dir)
        cfg = apply_params(params, base_config)

        cfg["data"]["train_data_df"] = train_csv
        cfg["data"]["eval_data_df"]  = eval_csv

        trial_tag = f"trial{trial.number:03d}_fold{test_fold}"
        cfg["train"]["checkpoint_path"] = os.path.join(tmp_dir, f"ckpt_{trial_tag}")
        cfg["train"]["log_path"]        = os.path.join(tmp_dir, f"log_{trial_tag}")
        cfg["model"]["model_name"]      = f"bayes_{trial_tag}"

        proxy_max_epochs = 60
        cfg["train"]["max_epochs"]          = proxy_max_epochs
        cfg["train"]["min_epochs"]          = 10
        cfg["train"]["earlystrop_patience"] = 10
        prod_max  = base_config["train"].get("max_epochs", 100)
        orig_warm = cfg["train"]["optimizer"].get("warm_epoch", 5)
        cfg["train"]["optimizer"]["warm_epoch"] = max(2, int(orig_warm * proxy_max_epochs / prod_max))

        rng_np    = np.random.get_state()
        rng_py    = random.getstate()
        rng_torch = torch.random.get_rng_state()
        rng_cuda  = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        try:
            metrics = train(cfg)
            if metrics:
                pcc  = float(metrics.get("PCC", 0.0))
                r2   = float(metrics.get("R2",  0.0))
                mse  = float(metrics.get("MSE", 0.0))
                mae  = float(metrics.get("MAE", 0.0))
                rmsd = float(np.sqrt(max(mse, 0.0)))
            else:
                pcc = r2 = mse = mae = rmsd = 0.0
        except Exception as e:
            import traceback
            print(f"  [Trial {trial.number}] fold {test_fold} ERROR: {e}")
            traceback.print_exc()
            pcc = r2 = mse = mae = rmsd = 0.0
        finally:
            try:
                np.random.set_state(rng_np)
                random.setstate(rng_py)
                torch.random.set_rng_state(rng_torch)
                if rng_cuda is not None:
                    torch.cuda.set_rng_state_all(rng_cuda)
            except Exception:
                pass

            ckpt_path = cfg["train"]["checkpoint_path"]
            if os.path.exists(ckpt_path):
                for f in os.listdir(ckpt_path):
                    if f == "last.ckpt":
                        os.remove(os.path.join(ckpt_path, f))

            try:
                import glob
                for shm_file in glob.glob('/dev/shm/torch_*'):
                    parts = os.path.basename(shm_file).split('_')
                    if len(parts) < 3:
                        continue
                    try:
                        pid = int(parts[1])
                    except ValueError:
                        continue
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        try:
                            os.remove(shm_file)
                        except OSError:
                            pass
            except Exception:
                pass

        pccs.append(pcc); r2s.append(r2); mses.append(mse); maes.append(mae); rmsds.append(rmsd)

        fold_csv = os.path.join(OUT_DIR, f"folds_gpu{gpu_id}.csv")
        fold_row = {
            "trial": trial.number, "fold": test_fold, "gpu": gpu_id,
            "PCC": pcc, "R2": r2, "MSE": mse, "MAE": mae, "RMSE": rmsd,
        }
        fold_row.update(params)
        pd.DataFrame([fold_row]).to_csv(
            fold_csv, mode='a', header=not os.path.exists(fold_csv), index=False)

        trial.report(float(np.mean(pccs)), step=step)
        if trial.should_prune():
            csv_path = os.path.join(OUT_DIR, f"trials_gpu{gpu_id}.csv")
            row = {"trial": trial.number, "gpu": gpu_id,
                   "status": "PRUNED", "pruned_at_fold": test_fold,
                   "mean_PCC": float(np.mean(pccs)),
                   "mean_R2":  float(np.mean(r2s)),
                   "mean_MSE": float(np.mean(mses)),
                   "mean_MAE": float(np.mean(maes)),
                   "mean_RMSE": float(np.mean(rmsds))}
            row.update(params)
            pd.DataFrame([row]).to_csv(
                csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
            raise optuna.TrialPruned()

    for idx, fold_id in enumerate(proxy_folds):
        trial.set_user_attr(f"fold{fold_id}_PCC", float(pccs[idx]))
        trial.set_user_attr(f"fold{fold_id}_MSE", float(mses[idx]))

    mean_pcc = float(np.mean(pccs))
    trial.set_user_attr("PCC",      mean_pcc)
    trial.set_user_attr("R2",       float(np.mean(r2s)))
    trial.set_user_attr("MSE",      float(np.mean(mses)))
    trial.set_user_attr("MAE",      float(np.mean(maes)))
    trial.set_user_attr("RMSD",     float(np.mean(rmsds)))
    trial.set_user_attr("mean_PCC", mean_pcc)

    csv_path = os.path.join(OUT_DIR, f"trials_gpu{gpu_id}.csv")
    row = {"trial": trial.number, "gpu": gpu_id,
           "mean_PCC": mean_pcc,
           "mean_R2":  float(np.mean(r2s)),
           "mean_MSE": float(np.mean(mses)),
           "mean_MAE": float(np.mean(maes)),
           "mean_RMSE": float(np.mean(rmsds))}
    for idx, fold_id in enumerate(proxy_folds):
        row[f"fold{fold_id}_PCC"] = float(pccs[idx])
        row[f"fold{fold_id}_MSE"] = float(mses[idx])
        row[f"fold{fold_id}_MAE"] = float(maes[idx])
        row[f"fold{fold_id}_R2"]  = float(r2s[idx])
        row[f"fold{fold_id}_RMSE"] = float(rmsds[idx])
    row.update(params)
    pd.DataFrame([row]).to_csv(
        csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    return mean_pcc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bayesian HPO for DLcatalysis")
    parser.add_argument("--config",       type=str, default=CONFIG_BASE)
    parser.add_argument("--n_trials",     type=int, default=30)
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--proxy_folds",  type=int, nargs="+",
                        default=list(range(10)),
                        help="Fold indices used as proxy test sets (default: all 10)")
    parser.add_argument("--study_name",   type=str, default="enzsub_hpo")
    parser.add_argument("--storage",      type=str, default=None,
                        help="Optuna storage URI for resumable runs")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = os.path.join(OUT_DIR, f"tmp_gpu{args.gpu}")
    os.makedirs(tmp_dir, exist_ok=True)

    base_config = init_config(args.config)
    base_config["train"]["n_gpu"] = [0]

    print(f"===== Bayesian HPO  GPU={args.gpu}  trials={args.n_trials} =====")
    print(f"Proxy folds : {args.proxy_folds}")
    print(f"Study       : {args.study_name}")
    print()

    pruner = optuna.pruners.NopPruner()

    if args.storage and args.storage.startswith("sqlite://"):
        storage = args.storage
    else:
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(args.storage)
        ) if args.storage else None

    n_existing  = 0
    n_completed = 0
    if storage is not None:
        try:
            tmp_study = optuna.load_study(study_name=args.study_name, storage=storage)
            n_existing  = len(tmp_study.trials)
            n_completed = sum(1 for t in tmp_study.trials if t.state.name == "COMPLETE")
        except Exception:
            n_existing = 0

    startup = 0 if n_completed >= 10 else max(3, 10 - n_completed)

    sampler = optuna.samplers.TPESampler(
        seed=42 + args.gpu * 1000 + n_existing * 97,
        n_startup_trials=startup,
        multivariate=True,
        group=True,
        constant_liar=True,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    print(f"[HPO] resume: {n_existing} total / {n_completed} completed, "
          f"startup_trials={startup}, seed_offset={n_existing * 97}")

    study.optimize(
        lambda trial: objective(trial, base_config, args.proxy_folds, tmp_dir),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best  = study.best_trial
    attrs = best.user_attrs
    print(f"\n{'='*40}")
    print(f"Best trial : #{best.number}")
    for m in ["PCC", "R2", "MSE", "MAE", "RMSD"]:
        print(f"  {m:<6}: {attrs.get(m, study.best_value if m=='PCC' else 0):.4f}")
    print(f"{'='*40}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print(f"\nWorker done. Aggregation handled by orchestrator.")
    print(f"Checkpoints preserved in: {tmp_dir}")


if __name__ == "__main__":
    main()
