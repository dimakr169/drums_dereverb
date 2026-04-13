from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_dict(v) for v in obj]
    return obj


def _validate_config(cfg: dict) -> None:
    required_top = ["experiment", "runtime", "data", "train", "model"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Missing top-level config sections: {missing}")

    backbone = cfg["model"].get("backbone", None)
    if backbone not in {"unet", "dit"}:
        raise ValueError("model.backbone must be either 'unet' or 'dit'.")

    residual_mode = cfg["train"].get("residual_mode", None)
    if residual_mode not in {"next_delta_norm", "direct"}:
        raise ValueError(
            "train.residual_mode must be 'next_delta_norm' or 'direct'."
        )

    if backbone == "dit" and residual_mode != "next_delta_norm":
        raise ValueError(
            "DiT currently supports only train.residual_mode='next_delta_norm'."
        )

    lr_policy = cfg["train"].get("lr_policy", None)
    valid_lr_policies = {
        "fixed",
        "cosine_restart",
        "warmup_cosine",
        "two_phase_cosine",
    }
    if lr_policy not in valid_lr_policies:
        raise ValueError(
            f"train.lr_policy must be one of {sorted(valid_lr_policies)}."
        )

    opt_name = cfg["train"].get("optimizer", {}).get("name", None)
    if opt_name not in {"adam", "adamw"}:
        raise ValueError("train.optimizer.name must be 'adam' or 'adamw'.")



def load_config(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    _validate_config(cfg_dict)

    cfg = _to_namespace(cfg_dict)
    cfg.config_path = str(path)
    return cfg


def config_to_dict(cfg) -> dict:
    out = _to_dict(cfg)
    if isinstance(out, dict):
        out.pop("config_path", None)
    return out


def save_config(cfg, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config_to_dict(cfg)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)