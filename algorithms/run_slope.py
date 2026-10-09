# -*- coding: utf-8 -*-
"""Unified CLI entry for slope-line extraction.

Usage:
    python run_slope.py --config config.json
See config.example.json for all tunable parameters.
"""
import sys, os, json, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slopeline_core import Config, run

def main():
    ap = argparse.ArgumentParser(description="Slope crest/toe line extraction from point cloud")
    ap.add_argument("--config", required=True, help="path to config json")
    ap.add_argument("--quiet", action="store_true", help="less console output")
    ap.add_argument("--auto-gap", choices=("on", "off"), default=None, help="override automatic gap recovery")
    args = ap.parse_args()
    if not os.path.exists(args.config):
        sys.exit(f"[error] config not found: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        d = json.load(f)
    base = os.path.dirname(os.path.abspath(args.config))
    cfg = Config.from_dict(d, base)
    if not cfg.input:
        sys.exit("[error] config must contain 'input'")
    if not cfg.output:
        sys.exit("[error] config must contain 'output'")
    if args.auto_gap is not None:
        cfg.auto_gap_enabled = (args.auto_gap == "on")
    print("== Slope line extraction ==")
    print(f"  input : {cfg.input}")
    print(f"  output: {cfg.output}")
    meta = run(cfg, verbose=not args.quiet)
    print("== done ==")
    print(f"  crest={meta['line_counts']['crest']}  toe={meta['line_counts']['toe']}  "
          f"auto_gap={meta['line_counts'].get('auto_gap', 0)}  "
          f"total_length={meta['total_length_m']}m")
    print(f"  files -> {cfg.output}")

if __name__ == "__main__":
    main()
