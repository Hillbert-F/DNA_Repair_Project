"""Command-line interface for config-driven IDEA runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import ConfigError, load_config
from .energy import calculate_energy_files
from .pipeline import IdeaPipeline
from .preprocess import PreprocessError, prepare_data_dir, prepare_data_from_config
from .strategy2 import Strategy2Error, Strategy2Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m idea.cli", description="Run IDEA with cached artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run training, testing, and energy for all jobs.")
    run_parser.add_argument("--config", required=True, help="Path to IDEA YAML config.")
    run_parser.add_argument("--force", action="store_true", help="Recompute artifacts even when cache entries exist.")
    run_parser.add_argument("--dry-run", action="store_true", help="Print resolved hashes without running.")

    train_parser = subparsers.add_parser("train", help="Produce or reuse one training gamma artifact.")
    train_parser.add_argument("--config", required=True, help="Path to IDEA YAML config.")
    train_parser.add_argument("--training-set", required=True, help="Training set name from config.")
    train_parser.add_argument("--force", action="store_true", help="Recompute the training artifact.")

    test_parser = subparsers.add_parser("test", help="Produce or reuse one testing phi artifact.")
    test_parser.add_argument("--config", required=True, help="Path to IDEA YAML config.")
    test_parser.add_argument("--testing-set", required=True, help="Testing set name from config.")
    test_parser.add_argument("--force", action="store_true", help="Recompute the testing artifact.")

    energy_parser = subparsers.add_parser("energy", help="Calculate energy from explicit gamma and phi paths.")
    energy_parser.add_argument("--gamma", required=True, help="Path to gamma file.")
    energy_parser.add_argument("--phi", required=True, help="Path to phi file.")
    energy_parser.add_argument("--out", required=True, help="Output directory.")

    prep_parser = subparsers.add_parser("prepare-data", help="Prepare raw PDBs into training/testing *_modified.pdb files.")
    prep_parser.add_argument("--data-dir", default="data", help="Directory containing raw .pdb/.pdb1/.ent files.")
    prep_parser.add_argument("--ids", nargs="*", default=None, help="Optional raw PDB ids/stems to process. Defaults to all raw PDB files in data-dir.")
    prep_parser.add_argument("--targets", nargs="+", default=["training", "testing"], choices=["training", "testing"], help="Where to copy prepared *_modified.pdb files.")
    prep_parser.add_argument("--dna-mode", default="auto", choices=["auto", "ss", "ds"], help="Raw structure DNA mode: auto detects one vs two DNA chains; ss keeps one DNA chain; ds keeps two DNA chains.")
    prep_parser.add_argument("--no-overwrite", action="store_true", help="Fail if an output *_modified.pdb already exists.")

    strategy2_parser = subparsers.add_parser("strategy2", help="Run Strategy2 ridge refinement and IDEA-style energy calculation.")
    strategy2_parser.add_argument("--config", required=True, help="Path to Strategy2 YAML config.")
    strategy2_parser.add_argument("--run", default=None, help="Optional Strategy2 run name to execute.")
    strategy2_parser.add_argument("--force", action="store_true", help="Recompute Strategy2 artifacts even when cache entries exist.")
    strategy2_parser.add_argument("--dry-run", action="store_true", help="Print Strategy2 cache keys without running.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "energy":
            energy_path = calculate_energy_files(args.gamma, args.phi, args.out)
            print(f"Energy written to: {energy_path}")
            return 0
        if args.command == "prepare-data":
            prepared = prepare_data_dir(
                args.data_dir,
                Path(__file__).resolve().parents[1],
                targets=args.targets,
                ids=args.ids,
                overwrite=not args.no_overwrite,
                dna_mode=args.dna_mode,
            )
            for item in prepared:
                print(f"Prepared {item.output_id}: kept={item.kept}")
                if item.training_path:
                    print(f"  training: {item.training_path}")
                if item.testing_path:
                    print(f"  testing:  {item.testing_path}")
            return 0

        if args.command == "strategy2":
            if not args.dry_run:
                _prepare_data_head_if_configured(args.config)
            config = load_config(args.config, require_training=False)
            pipeline = Strategy2Pipeline(config)
            if args.dry_run:
                print(json.dumps(pipeline.dry_run(run_name=args.run), indent=2, sort_keys=True))
                return 0
            experiment_dir = pipeline.run(run_name=args.run, force=args.force)
            print(f"Strategy2 experiment written to: {experiment_dir}")
            return 0

        if args.command == "run" and args.dry_run:
            config = load_config(args.config)
        else:
            _prepare_data_head_if_configured(args.config)
            config = load_config(args.config)
        pipeline = IdeaPipeline(config)
        if args.command == "run":
            if args.dry_run:
                print(json.dumps(pipeline.dry_run(), indent=2, sort_keys=True))
                return 0
            experiment_dir = pipeline.run_jobs(force=args.force)
            print(f"Experiment written to: {experiment_dir}")
            return 0
        if args.command == "train":
            artifact = pipeline.train(args.training_set, force=args.force)
            print(f"Training artifact: {artifact.path} ({'reused' if artifact.reused else 'created'})")
            return 0
        if args.command == "test":
            artifact = pipeline.test(args.testing_set, force=args.force)
            print(f"Testing artifact: {artifact.path} ({'reused' if artifact.reused else 'created'})")
            return 0
    except (ConfigError, PreprocessError, Strategy2Error, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 1


def _prepare_data_head_if_configured(config_path: str) -> None:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict) or not raw.get("data"):
        return
    raw["_config_dir"] = str(path.parent)
    prepared = prepare_data_from_config(raw, Path(__file__).resolve().parents[1])
    if prepared:
        print(f"[IDEA] Prepared {len(prepared)} raw PDB file(s) before pipeline run", flush=True)
        for item in prepared:
            outputs = []
            if item.training_path:
                outputs.append(f"training={item.training_path}")
            if item.testing_path:
                outputs.append(f"testing={item.testing_path}")
            print(f"[IDEA]   {item.output_id}: kept={item.kept}; " + "; ".join(outputs), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
