import argparse
import copy
import os
import sys
import torch
import gc
import pandas as pd
import numpy as np

PROJECT_ROOT = "../src"
sys.path.append(PROJECT_ROOT)
from tools.train_model import train
from util.tools import init_config


def run_cv(config, args):
    """Run 10-fold CV using fold_0.csv ~ fold_9.csv."""
    n_folds = args.n_folds
    fold_dir = args.fold_dir
    base_name = config['model']['model_name']
    base_ckpt = config['train']['checkpoint_path']
    base_log = config['train']['log_path']
    output_dir = args.output_dir or "cv_results"
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []

    for k in range(n_folds):
        train_folds = [i for i in range(n_folds) if i != k]

        # Merge train folds
        train_csv = os.path.join(output_dir, f"train_fold{k}.csv")
        dfs = [pd.read_csv(os.path.join(fold_dir, f"fold_{i}.csv")) for i in train_folds]
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(train_csv, index=False)

        # held-out fold is the evaluation set
        eval_csv = os.path.join(fold_dir, f"fold_{k}.csv")
        cfg = copy.deepcopy(config)
        cfg['data']['train_data_df'] = train_csv
        cfg['data']['eval_data_df']  = eval_csv
        cfg['model']['model_name'] = f"{base_name}_fold{k}"
        cfg['train']['checkpoint_path'] = os.path.join(base_ckpt, f"fold{k}")
        cfg['train']['log_path'] = os.path.join(base_log, f"fold{k}")
        cfg['train']['n_gpu'] = [0]

        best_metrics = train(cfg)

        if best_metrics:
            best_metrics['fold'] = k
            all_metrics.append(best_metrics)
            fold_csv = os.path.join(output_dir, f"fold_{k}.csv")
            pd.DataFrame([best_metrics]).to_csv(fold_csv, index=False)

        gc.collect()
        torch.cuda.empty_cache()

        # Clean /dev/shm torch files from dead worker processes
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

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        for col in ['PCC', 'R2', 'MSE', 'MAE', 'RMSE']:
            if col in df.columns:
                print(f"{col}: {df[col].mean():.4f} +/- {df[col].std():.4f}")
        summary_csv = os.path.join(output_dir, "summary.csv")
        df.to_csv(summary_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="../config/test.yml",
                       help='Path to config file')
    # 10-fold CV
    parser.add_argument('--cv', action='store_true', help='Run 10-fold CV (kept for compatibility; CV always runs)')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--fold_dir', type=str, default='../DataSet/final_data',
                       help='Directory containing fold_0.csv ~ fold_9.csv')
    parser.add_argument('--output_dir', type=str, help='Directory to save CV results')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    config = init_config(args.config)

    run_cv(config, args)
