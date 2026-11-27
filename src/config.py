"""Đọc config YAML + helper đường dẫn + tham số CLI dùng chung."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None, overrides=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, val = item.split("=", 1)
        val = yaml.safe_load(val)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return cfg


def p(cfg, key, *sub):
    """Trả về Path tuyệt đối: cfg['paths'][key] / sub..."""
    base = Path(cfg["paths"][key])
    if not base.is_absolute():
        base = ROOT / base
    return base.joinpath(*sub)


def add_common_args(parser):
    parser.add_argument("--config", default=None, help="đường dẫn config yaml (mặc định configs/config.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override config, ví dụ --set train.epochs=2")
    parser.add_argument("--data-path", default=None, help="override paths.raw_csv")
    return parser


def config_from_args(args):
    overrides = list(args.set)
    if getattr(args, "data_path", None):
        overrides.append(f"paths.raw_csv={args.data_path}")
    return load_config(args.config, overrides)
