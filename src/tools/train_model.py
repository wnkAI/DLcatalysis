import torch.multiprocessing; torch.multiprocessing.set_sharing_strategy("file_system")
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from util.tools import init_config, set_seed
from util.data_module import Singledataset
from model.enzsub import EnzSub

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pytorch_lightning.callbacks as pl_callbacks

from collections import OrderedDict

def config_flatten(config: dict, prefix: str = "", sep: str = ".") -> OrderedDict:
    flat_items = OrderedDict()
    for key, value in config.items():
        current_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict) and not isinstance(value, OrderedDict):
            nested_flat = config_flatten(value, prefix=current_key, sep=sep)
            flat_items.update(nested_flat)
        else:
            flat_items[current_key] = value
    return flat_items

def _build_callbacks(config, trial=None):
    """Create PyTorch Lightning callbacks for training."""
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=config["train"]["checkpoint_path"],
        filename=config["model"]["model_name"] + "-{epoch:02d}-{val_MSE:.4f}",
        monitor="val/MSE",
        mode="min",
        save_top_k=1,
        save_last=False
    )

    callbacks = [
        checkpoint_callback,
        pl_callbacks.EarlyStopping(
            monitor="val/MSE",
            patience=config['train']['earlystrop_patience'],
            mode="min",
            verbose=True
        ),
        pl_callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl_callbacks.GradientAccumulationScheduler(
            scheduling={0: config["train"].get("grad_accum", 4)}
        )
    ]

    if trial is not None:
        try:
            from optuna.integration import PyTorchLightningPruningCallback
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val/MSE"))
        except ImportError:
            pass

    return callbacks, checkpoint_callback


def _build_loggers(config):
    """Create TensorBoard and CSV loggers."""
    logger = TensorBoardLogger(
        save_dir=config["train"]["log_path"],
        name=config["model"]["model_name"],
        version=None,
        default_hp_metric=False
    )
    csv_logger = CSVLogger(
        save_dir=config["train"]["log_path"],
        name=config["model"]["model_name"],
    )
    logger.log_hyperparams(config_flatten(config))
    return [logger, csv_logger]


def _print_generalization_summary(val_r2, val_pcc, test_r2, test_pcc):
    print(f"test PCC={test_pcc:.4f} R2={test_r2:.4f}")


def train(config: dict, trial=None):

    set_seed(config["train"]["seed"])

    dataset = Singledataset(config)
    model = EnzSub(config)

    callbacks, checkpoint_callback = _build_callbacks(config, trial)
    loggers = _build_loggers(config)

    trainer = pl.Trainer(
        accelerator=config["train"]["device"],
        devices=config["train"]["n_gpu"],
        max_epochs=config['train']['max_epochs'],
        min_epochs=config['train']['min_epochs'],
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        val_check_interval=config['train']['valid_freq'],
        log_every_n_steps=50,
        enable_progress_bar=False,
        precision=config["train"].get("precision", "16-mixed"),
        strategy="auto",
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(model, datamodule=dataset)

    val_pcc = float(trainer.callback_metrics.get("val/PCC", 0))
    val_r2  = float(trainer.callback_metrics.get("val/R2",  0))

    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        trainer.test(model, datamodule=dataset, ckpt_path=best_ckpt_path)
    else:
        trainer.test(model, datamodule=dataset)

    # Prefer metrics stored directly on the model by on_test_epoch_end
    # (avoids Lightning callback_metrics quirks with sync_dist + epoch_end logging)
    direct = getattr(model, '_last_test_metrics', None)
    if direct:
        test_r2  = float(direct.get("R2",  0))
        test_pcc = float(direct.get("PCC", 0))
        test_scc = float(direct.get("SCC", 0))
        test_mse = float(direct.get("MSE", 0))
        test_mae = float(direct.get("MAE", 0))
    else:
        test_r2  = float(trainer.callback_metrics.get("test/R2",  0))
        test_pcc = float(trainer.callback_metrics.get("test/PCC", 0))
        test_scc = float(trainer.callback_metrics.get("test/SCC", 0))
        test_mse = float(trainer.callback_metrics.get("test/MSE", 0))
        test_mae = float(trainer.callback_metrics.get("test/MAE", 0))

    _print_generalization_summary(val_r2, val_pcc, test_r2, test_pcc)

    best_metrics = {
        "best_model_path": best_ckpt_path,
        "R2":      test_r2,
        "PCC":     test_pcc,
        "SCC":     test_scc,
        "MSE":     test_mse,
        "MAE":     test_mae,
        "val_PCC": val_pcc,
        "val_R2":  val_r2,
    }

    return best_metrics

if __name__ == "__main__":
    
    print("Test")
    config_fp = "../../config/test.yml"
    config = init_config(config_fp)
    print(config_flatten(config))