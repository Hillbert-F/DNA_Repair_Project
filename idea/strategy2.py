"""Strategy2 ridge-refinement workflow for IDEA.

Strategy2 fits a refined gamma directly from experimental binding signals and a
matching IDEA phi matrix, then calculates IDEA-style energies with that refined
gamma. It intentionally does not produce model-prediction outputs or LOO results.
"""

from __future__ import annotations

import copy
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .config import ConfigError, deep_merge, resolve_path
from .energy import ENERGY_CALCULATION_VERSION, calculate_energy_files, load_phi
from .hashing import file_sha256, stable_hash
from .pipeline import Artifact, IdeaPipeline, read_sequence_lines


STRATEGY2_VERSION = "idea-strategy2-v1"
SUPPORTED_TARGET_SIGNS = {"negate", "identity"}
DEFAULT_STRATEGY2_SETTINGS = {
    "alpha": 1.0,
    "alpha_grid": {"min_exp": -4.0, "max_exp": 2.0, "num": 50},
    "fit_intercept": True,
    "target_sign": "negate",
}


@dataclass(frozen=True)
class Strategy2RunResult:
    name: str
    fit_phi_artifact: Artifact
    ridge_artifact: Artifact
    energy_artifact: Artifact


class Strategy2Error(RuntimeError):
    """Raised when Strategy2 fitting or energy calculation fails."""


class Strategy2Pipeline:
    def __init__(self, config: dict[str, Any], repo_root: str | Path | None = None):
        self.config = config
        self.repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[1]
        self.runs_root = self.repo_root / "runs_strategy2"
        self.cache_root = self.runs_root / "cache"
        self.experiments_root = self.runs_root / "experiments"
        self.idea = IdeaPipeline(config, repo_root=self.repo_root, runs_root=self.runs_root)
        self.runs = self._resolve_strategy2_runs()

    def dry_run(self, run_name: str | None = None) -> dict[str, Any]:
        selected = self._selected_runs(run_name)
        runs = []
        for name, spec in selected.items():
            fit_name, fit_spec = self._fit_testing_spec(name, spec)
            fit_key = self._testing_key_for_spec(fit_name, fit_spec)
            ridge_key = self._ridge_key(name, spec, fit_key)
            energy_key = self._strategy2_energy_key(ridge_key, fit_key)
            runs.append(
                {
                    "name": name,
                    "template_testing": spec["template_testing"],
                    "sequences": spec["sequences"]["path"],
                    "exp": spec["exp"]["path"],
                    "alpha": spec["alpha"],
                    "target_sign": spec["target_sign"],
                    "fit_intercept": spec["fit_intercept"],
                    "fit_phi_hash": fit_key,
                    "ridge_hash": ridge_key,
                    "energy_hash": energy_key,
                    "fit_phi_artifact": str(self._testing_cache_dir(fit_name, fit_key).relative_to(self.repo_root)),
                    "ridge_artifact": str(self._cache_dir("ridge", ridge_key, name).relative_to(self.repo_root)),
                    "energy_artifact": str(self._cache_dir("energy", energy_key, name).relative_to(self.repo_root)),
                }
            )
        return {"runs_root": str(self.runs_root.relative_to(self.repo_root)), "strategy2_runs": runs}

    def run(self, run_name: str | None = None, force: bool = False) -> Path:
        selected = self._selected_runs(run_name)
        experiment_dir = self.experiments_root / str(self.config["run_id"])
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with (experiment_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_public_config(self.config), handle, sort_keys=False)

        rows = []
        self._info(f"Starting Strategy2 run '{self.config['run_id']}' with {len(selected)} job(s)")
        for index, (name, spec) in enumerate(selected.items(), start=1):
            self._info(f"Strategy2 job {index}/{len(selected)}: {name}")
            result = self.run_one(name, spec, force=force)

            job_dir = experiment_dir / _safe_name(name)
            job_dir.mkdir(exist_ok=True)
            (job_dir / "fit_phi_artifact.txt").write_text(str(result.fit_phi_artifact.path) + "\n", encoding="utf-8")
            (job_dir / "ridge_artifact.txt").write_text(str(result.ridge_artifact.path) + "\n", encoding="utf-8")
            (job_dir / "energy_artifact.txt").write_text(str(result.energy_artifact.path) + "\n", encoding="utf-8")

            rows.append(
                {
                    "job": name,
                    "template_testing": spec["template_testing"],
                    "fit_phi_hash": result.fit_phi_artifact.key,
                    "ridge_hash": result.ridge_artifact.key,
                    "energy_hash": result.energy_artifact.key,
                    "fit_phi_reused": result.fit_phi_artifact.reused,
                    "ridge_reused": result.ridge_artifact.reused,
                    "energy_reused": result.energy_artifact.reused,
                    "fit_phi_artifact_path": result.fit_phi_artifact.path,
                    "ridge_artifact_path": result.ridge_artifact.path,
                    "energy_artifact_path": result.energy_artifact.path,
                    "energy_path": result.energy_artifact.path / "Energy_mg.txt",
                }
            )

        with (experiment_dir / "jobs_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "job",
                "template_testing",
                "fit_phi_hash",
                "ridge_hash",
                "energy_hash",
                "fit_phi_reused",
                "ridge_reused",
                "energy_reused",
                "fit_phi_artifact_path",
                "ridge_artifact_path",
                "energy_artifact_path",
                "energy_path",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return experiment_dir

    def run_one(self, name: str, spec: dict[str, Any], force: bool = False) -> Strategy2RunResult:
        fit_phi_artifact = self.fit_phi(name, spec, force=force)
        ridge_artifact = self.fit_ridge(name, spec, fit_phi_artifact, force=force)
        energy_artifact = self.energy_from_ridge(name, spec, ridge_artifact, fit_phi_artifact, force=force)
        return Strategy2RunResult(name, fit_phi_artifact, ridge_artifact, energy_artifact)

    def fit_phi(self, name: str, spec: dict[str, Any], force: bool = False) -> Artifact:
        fit_name, fit_spec = self._fit_testing_spec(name, spec)
        config = copy.deepcopy(self.config)
        config.setdefault("testing_sets", {})[fit_name] = fit_spec
        pipeline = IdeaPipeline(config, repo_root=self.repo_root, runs_root=self.runs_root)
        return pipeline.test(fit_name, force=force)

    def fit_ridge(self, name: str, spec: dict[str, Any], fit_phi_artifact: Artifact, force: bool = False) -> Artifact:
        key = self._ridge_key(name, spec, fit_phi_artifact.key)
        artifact_dir = self._cache_dir("ridge", key, name)
        gamma_path = artifact_dir / "gamma_refined.txt"
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists() and gamma_path.exists() and not force:
            self._info(f"Ridge cache hit: {name} -> {artifact_dir.relative_to(self.repo_root)}")
            return Artifact("strategy2_ridge", key, artifact_dir, manifest_path, reused=True)

        self._info(f"Ridge cache miss: {name} -> {artifact_dir.relative_to(self.repo_root)}")
        if force and artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        phi_path = fit_phi_artifact.path / "phi_decoys.txt"
        exp_path = Path(spec["exp"]["path"])
        phi = load_phi(phi_path)
        exp = load_exp(exp_path)
        if phi.shape[0] != exp.shape[0]:
            raise Strategy2Error(
                f"Strategy2 row mismatch for '{name}': phi has {phi.shape[0]} rows, "
                f"exp has {exp.shape[0]} values."
            )
        fit = fit_ridge_gamma(phi, exp, spec)
        gamma = fit["gamma"]
        _np().savetxt(gamma_path, gamma.reshape(-1, 1), fmt="%.10f")

        manifest = {
            "kind": "strategy2_ridge",
            "strategy2_version": STRATEGY2_VERSION,
            "idea_version": __version__,
            "hash": key,
            "label": name,
            "template_testing": spec["template_testing"],
            "fit_phi_hash": fit_phi_artifact.key,
            "fit_phi": str(phi_path.relative_to(self.repo_root)),
            "gamma_refined": str(gamma_path.relative_to(self.repo_root)),
            "exp": spec["exp"],
            "sequences": spec["sequences"],
            "alpha": spec["alpha"],
            "alpha_used": fit["alpha_used"],
            "alpha_grid": spec["alpha_grid"],
            "fit_intercept": spec["fit_intercept"],
            "intercept": fit["intercept"],
            "target_sign": spec["target_sign"],
            "n_samples": int(phi.shape[0]),
            "n_features": int(phi.shape[1]),
        }
        _write_json(manifest_path, manifest)
        return Artifact("strategy2_ridge", key, artifact_dir, manifest_path, reused=False)

    def energy_from_ridge(
        self,
        name: str,
        spec: dict[str, Any],
        ridge_artifact: Artifact,
        fit_phi_artifact: Artifact,
        force: bool = False,
    ) -> Artifact:
        key = self._strategy2_energy_key(ridge_artifact.key, fit_phi_artifact.key)
        artifact_dir = self._cache_dir("energy", key, name)
        energy_path = artifact_dir / "Energy_mg.txt"
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists() and energy_path.exists() and not force:
            self._info(f"Strategy2 energy cache hit: {name} -> {artifact_dir.relative_to(self.repo_root)}")
            return Artifact("strategy2_energy", key, artifact_dir, manifest_path, reused=True)

        self._info(f"Strategy2 energy cache miss: {name} -> {artifact_dir.relative_to(self.repo_root)}")
        if force and artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        calculate_energy_files(
            ridge_artifact.path / "gamma_refined.txt",
            fit_phi_artifact.path / "phi_decoys.txt",
            artifact_dir,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "kind": "strategy2_energy",
                "strategy2_version": STRATEGY2_VERSION,
                "hash": key,
                "label": name,
                "ridge_hash": ridge_artifact.key,
                "fit_phi_hash": fit_phi_artifact.key,
                "template_testing": spec["template_testing"],
            }
        )
        _write_json(manifest_path, manifest)
        return Artifact("strategy2_energy", key, artifact_dir, manifest_path, reused=False)

    def _resolve_strategy2_runs(self) -> dict[str, dict[str, Any]]:
        raw = self.config.get("strategy2")
        if not raw:
            raise ConfigError("No strategy2 runs configured. Add a top-level 'strategy2:' block.")
        if not isinstance(raw, dict):
            raise ConfigError("strategy2 must be a mapping.")
        raw_defaults = raw.get("defaults", {})
        defaults = deep_merge(DEFAULT_STRATEGY2_SETTINGS, raw_defaults if isinstance(raw_defaults, dict) else None)
        if raw_defaults and not isinstance(raw_defaults, dict):
            raise ConfigError("strategy2.defaults must be a mapping when provided.")
        raw_runs = raw.get("runs")
        if raw_runs is None:
            raw_runs = {key: value for key, value in raw.items() if key != "defaults"}
        if not isinstance(raw_runs, dict) or not raw_runs:
            raise ConfigError("strategy2.runs must define at least one run.")

        resolved: dict[str, dict[str, Any]] = {}
        for name, run_spec in raw_runs.items():
            if not isinstance(run_spec, dict):
                raise ConfigError(f"strategy2 run '{name}' must be a mapping.")
            spec = deep_merge(defaults, run_spec)
            resolved[str(name)] = self._resolve_one_run(str(name), spec)
        return resolved

    def _resolve_one_run(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        template_testing = _require_nonempty(spec, "template_testing", f"strategy2.runs.{name}")
        if template_testing not in self.config.get("testing_sets", {}):
            raise ConfigError(f"strategy2 run '{name}' references unknown testing set '{template_testing}'.")

        seq_path = resolve_path(self.config, _require_nonempty(spec, "sequences", f"strategy2.runs.{name}"))
        exp_path = resolve_path(self.config, _require_nonempty(spec, "exp", f"strategy2.runs.{name}"))
        if not seq_path.exists():
            raise ConfigError(f"Strategy2 sequence file does not exist: {seq_path}")
        if not exp_path.exists():
            raise ConfigError(f"Strategy2 exp file does not exist: {exp_path}")
        seqs = read_sequence_lines(seq_path)
        exp = load_exp(exp_path)
        if len(seqs) != exp.shape[0]:
            raise ConfigError(
                f"strategy2 run '{name}' expects one exp value per sequence: "
                f"{seq_path} has {len(seqs)} sequences, {exp_path} has {exp.shape[0]} values."
            )

        target_sign = str(spec.get("target_sign", "negate")).strip().lower()
        if target_sign not in SUPPORTED_TARGET_SIGNS:
            raise ConfigError(f"strategy2 run '{name}' target_sign must be one of {sorted(SUPPORTED_TARGET_SIGNS)}.")
        alpha = _normalize_alpha(spec.get("alpha", 1.0), f"strategy2.runs.{name}.alpha")
        alpha_grid = _normalize_alpha_grid(spec.get("alpha_grid", DEFAULT_STRATEGY2_SETTINGS["alpha_grid"]), name)
        fit_intercept = _to_bool(spec.get("fit_intercept", True), f"strategy2.runs.{name}.fit_intercept")

        return {
            "name": name,
            "template_testing": template_testing,
            "sequences": {"path": str(seq_path), "sha256": file_sha256(seq_path), "n_sequences": len(seqs)},
            "exp": {"path": str(exp_path), "sha256": file_sha256(exp_path), "n_values": int(exp.shape[0])},
            "alpha": alpha,
            "alpha_grid": alpha_grid,
            "fit_intercept": fit_intercept,
            "target_sign": target_sign,
        }

    def _selected_runs(self, run_name: str | None) -> dict[str, dict[str, Any]]:
        if run_name is None:
            return self.runs
        if run_name not in self.runs:
            raise ConfigError(f"Unknown strategy2 run: {run_name}")
        return {run_name: self.runs[run_name]}

    def _fit_testing_spec(self, name: str, spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        base = copy.deepcopy(self.config["testing_sets"][spec["template_testing"]])
        base["sequences"] = spec["sequences"]["path"]
        return _safe_name(f"fit_{name}"), base

    def _testing_key_for_spec(self, name: str, spec: dict[str, Any]) -> str:
        config = copy.deepcopy(self.config)
        config.setdefault("testing_sets", {})[name] = spec
        return IdeaPipeline(config, repo_root=self.repo_root, runs_root=self.runs_root).testing_key(name, spec)

    def _ridge_key(self, name: str, spec: dict[str, Any], fit_phi_key: str) -> str:
        return stable_hash(
            {
                "kind": "strategy2_ridge",
                "version": STRATEGY2_VERSION,
                "name": name,
                "template_testing": spec["template_testing"],
                "fit_phi_hash": fit_phi_key,
                "sequences": spec["sequences"],
                "exp": spec["exp"],
                "alpha": spec["alpha"],
                "alpha_grid": spec["alpha_grid"],
                "fit_intercept": spec["fit_intercept"],
                "target_sign": spec["target_sign"],
            }
        )

    def _strategy2_energy_key(self, ridge_key: str, fit_phi_key: str) -> str:
        return stable_hash(
            {
                "kind": "strategy2_energy",
                "version": STRATEGY2_VERSION,
                "energy_version": ENERGY_CALCULATION_VERSION,
                "ridge_hash": ridge_key,
                "fit_phi_hash": fit_phi_key,
            }
        )

    def _testing_cache_dir(self, label: str, key: str) -> Path:
        return self.runs_root / "cache" / "testing" / f"{_safe_name(label)}__{key}"

    def _cache_dir(self, kind: str, key: str, label: str) -> Path:
        kind_dir = self.cache_root / kind
        path = kind_dir / f"{_safe_name(label)}__{key}"
        if not path.exists() and kind_dir.exists():
            matches = sorted(kind_dir.glob(f"*__{key}"))
            if matches:
                return matches[0]
        return path

    def _info(self, message: str) -> None:
        print(f"[IDEA] {message}", flush=True)


def load_exp(path: str | Path):
    np = _np()
    values = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 1:
                raise Strategy2Error(f"Expected one numeric exp value per line in {path}, line {lineno}: {stripped}")
            try:
                values.append(float(parts[0]))
            except ValueError as exc:
                raise Strategy2Error(f"Invalid numeric exp value in {path}, line {lineno}: {stripped}") from exc
    if not values:
        raise Strategy2Error(f"Exp file is empty: {path}")
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise Strategy2Error(f"Exp file contains NaN or inf: {path}")
    return arr


def fit_ridge_gamma(phi, exp, spec: dict[str, Any]) -> dict[str, Any]:
    np = _np()
    from sklearn.linear_model import Ridge, RidgeCV

    phi = np.asarray(phi, dtype=float)
    exp = np.asarray(exp, dtype=float).reshape(-1)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    if phi.shape[0] != exp.shape[0]:
        raise Strategy2Error(f"Ridge row mismatch: phi has {phi.shape[0]} rows, exp has {exp.shape[0]} values.")
    if phi.shape[0] < 2:
        raise Strategy2Error("Strategy2 ridge fitting requires at least 2 sequence/exp pairs.")
    if np.isnan(phi).any() or np.isinf(phi).any():
        raise Strategy2Error("Phi matrix contains NaN or inf.")

    y = -exp if spec["target_sign"] == "negate" else exp
    fit_intercept = bool(spec["fit_intercept"])
    alpha = spec["alpha"]
    if alpha == "auto":
        grid = spec["alpha_grid"]
        alphas = np.logspace(float(grid["min_exp"]), float(grid["max_exp"]), int(grid["num"]))
        cv_folds = min(5, phi.shape[0])
        if cv_folds < 2:
            raise Strategy2Error("Strategy2 alpha=auto requires at least 2 samples for RidgeCV.")
        model = RidgeCV(alphas=alphas, cv=cv_folds, scoring="neg_mean_squared_error", fit_intercept=fit_intercept)
        model.fit(phi, y)
        alpha_used = float(model.alpha_)
    else:
        model = Ridge(alpha=float(alpha), fit_intercept=fit_intercept)
        model.fit(phi, y)
        alpha_used = float(alpha)

    gamma = np.asarray(model.coef_, dtype=float).reshape(-1)
    intercept = float(model.intercept_) if fit_intercept else 0.0
    if gamma.shape[0] != phi.shape[1]:
        raise Strategy2Error(f"Ridge gamma dimension {gamma.shape[0]} does not match phi columns {phi.shape[1]}.")
    return {"gamma": gamma, "alpha_used": alpha_used, "intercept": intercept}


def _normalize_alpha(value: Any, label: str) -> str | float:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a non-negative number or 'auto'.") from exc
    if alpha < 0:
        raise ConfigError(f"{label} must be non-negative.")
    return alpha


def _normalize_alpha_grid(value: Any, run_name: str) -> dict[str, float | int]:
    if not isinstance(value, dict):
        raise ConfigError(f"strategy2 run '{run_name}' alpha_grid must be a mapping.")
    try:
        min_exp = float(value.get("min_exp", -4.0))
        max_exp = float(value.get("max_exp", 2.0))
        num = int(value.get("num", 50))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"strategy2 run '{run_name}' alpha_grid values must be numeric.") from exc
    if num < 2:
        raise ConfigError(f"strategy2 run '{run_name}' alpha_grid.num must be at least 2.")
    if max_exp <= min_exp:
        raise ConfigError(f"strategy2 run '{run_name}' alpha_grid.max_exp must be greater than min_exp.")
    return {"min_exp": min_exp, "max_exp": max_exp, "num": num}


def _to_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ConfigError(f"{label} must be true or false.")


def _require_nonempty(mapping: dict[str, Any], key: str, label: str) -> str:
    if key not in mapping or mapping[key] in (None, ""):
        raise ConfigError(f"Missing required '{key}' in {label}.")
    return str(mapping[key])


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "strategy2"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _np():
    import numpy as np

    return np
