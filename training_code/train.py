#!/usr/bin/env python3
import os
import sys
import re
import json
from pathlib import Path

import torch
from ds_dict import ds_paths
from whisper_module import LoRAStreamedWhisper
from training_code.utils import Config, parse_cmdl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

# Add repo root to PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from careless_whisper_stream import load_streaming_model


SEED = 3407
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)
torch.set_float32_matmul_precision("high")

logs_root = f"{os.environ.get('HOME')}/ma/data/models/logs"
ckpt_root = f"{os.environ.get('HOME')}/ma/data/models/ckpts"

whisper_lrs: dict[str, float] = {
    'tiny': 1.5e-3,
    'base': 1e-3,
    'small': 5e-4,
    'medium': 2.5e-4,
    'large': 1.75e-4,
    'large-v2': 2e-4
}

project_names = {
    "lora": "LoRA_whisper_stream",
}


def _extract_epoch_from_name(path: Path) -> int:
    m = re.fullmatch(r"checkpoint-epoch=(-?\d+)\.ckpt", path.name)
    if m:
        return int(m.group(1))
    return -10**9


def _find_best_warmstart_checkpoint(run_name: str) -> str:
    """
    Looks in:
        {ckpt_root}/{run_name}/checkpoint/

    Accepted filename style ONLY:
    - checkpoint-epoch=XXXX.ckpt
    - checkpoint-epoch=-001.ckpt

    Preference order:
    1. best_model_path from a Lightning callback state in the checkpoint
    2. numerically highest checkpoint-epoch=XXXX.ckpt
    """
    run_dir = Path(ckpt_root) / run_name
    checkpoint_dir = run_dir / "checkpoint"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Warmstart checkpoint directory not found: {checkpoint_dir}"
        )

    ckpt_files = [
        p for p in checkpoint_dir.iterdir()
        if p.is_file() and re.fullmatch(r"checkpoint-epoch=-?\d+\.ckpt", p.name)
    ]
    if not ckpt_files:
        raise FileNotFoundError(
            f"No valid checkpoint files found in: {checkpoint_dir}"
        )

    # 1) Try to recover best_model_path from any saved Lightning checkpoint metadata
    for ckpt_path in sorted(ckpt_files):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            callbacks = ckpt.get("callbacks", {})
            for _, cb_state in callbacks.items():
                if isinstance(cb_state, dict):
                    best_model_path = cb_state.get("best_model_path")
                    if best_model_path and os.path.exists(best_model_path):
                        best_path = Path(best_model_path)
                        if re.fullmatch(r"checkpoint-epoch=-?\d+\.ckpt", best_path.name):
                            print(f"Using best warmstart checkpoint from metadata: {best_model_path}")
                            return str(best_path)
        except Exception:
            pass

    # 2) Fall back to highest checkpoint epoch
    best_fallback = max(ckpt_files, key=_extract_epoch_from_name)
    print(f"Using latest warmstart checkpoint by epoch fallback: {best_fallback}")
    return str(best_fallback)


def _apply_warmstart(model: LoRAStreamedWhisper, cfg: Config, model_name: str) -> None:
    warm_ckpt_path = _find_best_warmstart_checkpoint(cfg.warmstart)

    warm_model = load_streaming_model(
        name=model_name,
        gran=cfg.gran,
        multilingual=cfg.multilingual,
        device="cpu",
        local_ckpt_path=warm_ckpt_path,
    )

    missing, unexpected = model.model.load_state_dict(
        warm_model.state_dict(),
        strict=False,
    )

    print(f"Warmstarted self.model from {warm_ckpt_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def _save_untrained_checkpoint(model: LoRAStreamedWhisper, check_output_dir: str, cfg: Config) -> Path:
    checkpoint_dir = Path(check_output_dir) / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_path = checkpoint_dir / "checkpoint-epoch=-001.ckpt"

    checkpoint = {
        "state_dict": model.model.state_dict(),
        "hyper_parameters": {
            "size": cfg.size,
            "enc_emb_gran": cfg.gran,
            "gran": cfg.gran,
            "enc_context": cfg.extra_gran_blocks,
            "extra_gran_blocks": cfg.extra_gran_blocks,
            "rank": cfg.rank,
            "multilingual": cfg.multilingual,
        },
        "cfg": vars(cfg),
    }

    if hasattr(model.model, "dims"):
        checkpoint["dims"] = vars(model.model.dims)

    torch.save(checkpoint, save_path)
    print(f"Saved untrained checkpoint to: {save_path}")
    return save_path


def train_model(log_output_dir, check_output_dir, model_name, train_set, val_set, train_name, project_name, cfg: Config) -> None:
    Path(log_output_dir).mkdir(exist_ok=True)
    Path(check_output_dir).mkdir(exist_ok=True)

    config_path = Path(check_output_dir) / "cfg.json"
    with open(config_path, "w") as f:
        json.dump(vars(cfg), f, indent=4)

    wandblogger = WandbLogger(
        save_dir=log_output_dir,
        name=train_name,
        project=project_names[project_name]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=cfg.top_k,
        monitor="val/wer",
        mode="min",
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

    if cfg.early_stop:
        early_stop_callback = EarlyStopping(
            monitor="val/wer",
            min_delta=0.00,
            patience=2,
            mode="min",
        )
        callback_list.append(early_stop_callback)

    if cfg.lora and cfg.streaming_train:
        model = LoRAStreamedWhisper(
            cfg,
            model_name,
            cfg.lang,
            train_set,
            val_set,
            rank=cfg.rank,
            enc_emb_gran=cfg.gran,
            enc_context=cfg.extra_gran_blocks,
            sim_stream=cfg.sim_stream,
            calc_rwer_arwer=cfg.extra_eval
        )
    else:
        raise ValueError("Only lora + streaming_train path is implemented in this script.")

    # Warmstart weights only
    if cfg.warmstart not in (None, ""):
        print(f"Warmstarting from run: {cfg.warmstart}")
        _apply_warmstart(model, cfg, model_name)

    # Save freshly initialized / optionally warmstarted model and exit
    if getattr(cfg, "save_untrained", False):
        _save_untrained_checkpoint(model, check_output_dir, cfg)
        print("Saved untrained checkpoint only. Exiting without training.")
        return

    trainer = Trainer(
        accelerator=DEVICE,
        max_epochs=cfg.num_train_epochs,
        callbacks=callback_list,
        logger=wandblogger if not cfg.no_logger else False,
        deterministic=True,
        num_sanity_val_steps=1,
        strategy=cfg.strategy,
        fast_dev_run=cfg.fast_dev_run,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.gradient_accumulation_steps
    )

    # True resume has priority over warmstart
    if cfg.ckpt is None:
        print("Running full validation before training...")
        trainer.validate(model)
        trainer.fit(model)
    else:
        print("Running full validation before resumed training...")
        trainer.validate(model, ckpt_path=cfg.ckpt)
        trainer.fit(model, ckpt_path=cfg.ckpt)


if __name__ == "__main__":
    project_name = None
    args = parse_cmdl()

    cfg = Config(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        num_worker=args.num_worker,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gacc,
        no_logger=args.no_logger,
        dataset=args.dataset,
        name=args.name,
        top_k=args.top_k,
        sample_rate=16_000,
        ckpt=args.ckpt,
        size=args.size,
        lora=args.lora,
        lora_ckpt=args.lora_ckpt,
        rank=args.rank,
        gran=args.gran,
        extra_gran_blocks=args.extra_gran_blocks,
        sim_stream=args.simulate_stream,
        fast_dev_run=args.fast_dev_run,
        early_stop=args.early_stop,
        strategy=args.strategy,
        streaming_train=args.streaming_train,
        streaming_random=args.streaming_random,
        streaming_fraction=args.streaming_fraction,
        seed=SEED,
        multilingual=args.multilingual,
        custom_len=args.custom_len,
        precision=args.precision,
        precomputed_features=args.precomputed_features,
        warmstart=args.warmstart,
        extra_eval=args.extra_eval,
        save_untrained=args.save_untrained,
    )

    if cfg.streaming_train:
        assert cfg.sim_stream == cfg.streaming_train, "When running in full stream mode you must simulate streaming!"
        cfg.sim_stream = True

    dir_name = cfg.name
    project_name = "lora" if (cfg.lora and cfg.streaming_train) else None

    selected_train = (
        ds_paths[args.dataset]["precomputed"]["train"]
        if args.precomputed_features
        else ds_paths[args.dataset]["train"]
    )

    selected_val = (
        ds_paths[args.dataset]["precomputed"]["val"]
        if args.precomputed_features
        else ds_paths[args.dataset]["val"]
    )

    train_model(
        log_output_dir=os.path.join(logs_root, dir_name),
        check_output_dir=os.path.join(ckpt_root, dir_name),
        model_name=args.size,
        train_set=selected_train,
        val_set=selected_val,
        train_name=dir_name,
        project_name=project_name,
        cfg=cfg
    )