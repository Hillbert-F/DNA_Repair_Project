"""Configuration loading and validation for IDEA runs."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_DNA_MODES = {"ss", "ds"}
DEFAULTS = {
    "protein_chain": "A",
    "contact_cutoff_nm": 1.2,
    "gamma_cutoff_mode": 60,
    "phi": {
        "name": "phi_pairwise_contact_well",
        "params": [-8.0, 8.0, 0.7, 10],
    },
    "training_decoys": {
        "dna": 1000,
        "protein": 10000,
    },
    "max_testing_decoys": 1000000,
}


class ConfigError(ValueError):
    """Raised when a user config is invalid."""


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, require_training: bool = True) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping.")

    config = normalize_config(raw, config_path)
    validate_config(config, require_training=require_training)
    return config


def normalize_config(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Expand the simple user config into the internal advanced shape."""
    config = copy.deepcopy(raw)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    config["_repo_root"] = str(Path(__file__).resolve().parents[1])

    user_defaults = copy.deepcopy(raw.get("defaults", {}))
    if "dna_mode" in user_defaults:
        raise ConfigError("defaults.dna_mode is no longer supported; set dna_mode under each testing entry.")
    config["defaults"] = deep_merge(DEFAULTS, user_defaults)

    if not config.get("run_id"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config["run_id"] = f"{config_path.stem}_{stamp}"
        config["_run_id_auto_generated"] = True
    else:
        config["_run_id_auto_generated"] = False

    if "training_sets" not in config and "training" in config:
        training_spec = _normalize_simple_training(config, config["training"])
        training_name = _auto_training_name(training_spec["pdbs"])
        config["training_sets"] = {training_name: training_spec}
        config["_auto_training_name"] = training_name

    if "testing_sets" not in config and "testing" in config:
        testing_items = config["testing"]
        if isinstance(testing_items, dict):
            testing_items = [testing_items]
        if not isinstance(testing_items, list) or not testing_items:
            raise ConfigError("'testing' must be a non-empty mapping or list.")
        testing_sets: dict[str, Any] = {}
        for item in testing_items:
            testing_spec = _normalize_simple_testing(config, item)
            testing_name = _unique_name(_auto_testing_name(testing_spec), testing_sets)
            testing_sets[testing_name] = testing_spec
        config["testing_sets"] = testing_sets

    config.setdefault("training_sets", {})
    config.setdefault("testing_sets", {})

    if "jobs" not in config or config.get("jobs") is None:
        jobs = []
        for training_name in config["training_sets"]:
            for testing_name in config["testing_sets"]:
                jobs.append(
                    {
                        "name": _safe_name(f"{training_name}__{testing_name}"),
                        "training": training_name,
                        "testing": testing_name,
                    }
                )
        config["jobs"] = jobs
        config["_jobs_auto_generated"] = True
    else:
        config["_jobs_auto_generated"] = False

    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    config_relative = (Path(config["_config_dir"]) / path).resolve()
    repo_relative = (Path(config["_repo_root"]) / path).resolve()

    # User-provided relative paths can live beside the config file. Built-in IDEA
    # paths such as training/PDBs/... and testing/PDBs/... should resolve from
    # the repository root, not configs/. Prefer whichever candidate exists.
    if config_relative.exists():
        return config_relative
    if repo_relative.exists():
        return repo_relative
    if path.parts and path.parts[0] in {"training", "testing", "energy_calculation"}:
        return repo_relative
    return config_relative


def infer_path(config: dict[str, Any], kind: str, item_id: str) -> str:
    if kind == "training_pdb":
        rel = Path("training") / "PDBs" / f"{item_id}_modified.pdb"
    elif kind == "testing_pdb":
        rel = Path("testing") / "PDBs" / f"{item_id}_modified.pdb"
    elif kind == "testing_sequences":
        rel = Path("testing") / "sequences" / "dna_half.seq"
    else:
        raise ValueError(f"Unsupported path kind: {kind}")
    return rel.as_posix()


def phi_params_string(config: dict[str, Any], override: dict[str, Any] | None = None) -> str:
    phi = copy.deepcopy(config["defaults"]["phi"])
    if override and isinstance(override.get("phi"), dict):
        phi = deep_merge(phi, override["phi"])
    return "_".join(str(x) for x in phi["params"])


def validate_config(config: dict[str, Any], require_training: bool = True) -> None:
    defaults = config["defaults"]
    _validate_default_settings(defaults, "defaults")

    training_sets = config.get("training_sets", {})
    testing_sets = config.get("testing_sets", {})
    if not isinstance(training_sets, dict):
        raise ConfigError("'training_sets' must be a mapping.")
    if not isinstance(testing_sets, dict):
        raise ConfigError("'testing_sets' must be a mapping.")
    if require_training and not training_sets:
        raise ConfigError("No training sets were configured. Use 'training:' or 'training_sets:'.")
    if not testing_sets:
        raise ConfigError("No testing sets were configured. Use 'testing:' or 'testing_sets:'.")

    for name, training in training_sets.items():
        if not isinstance(training, dict):
            raise ConfigError(f"Training set '{name}' must be a mapping.")
        _validate_default_settings(deep_merge(defaults, training), f"training_sets.{name}")
        pdbs = training.get("pdbs")
        if not isinstance(pdbs, list) or not pdbs:
            raise ConfigError(f"Training set '{name}' must define a non-empty 'pdbs' list.")
        seen_ids: set[str] = set()
        for pdb in pdbs:
            pdb_id = _require_nonempty(pdb, "id", f"training_sets.{name}.pdbs")
            if pdb_id in seen_ids:
                raise ConfigError(f"Training set '{name}' has duplicate PDB id '{pdb_id}'.")
            seen_ids.add(pdb_id)
            path = resolve_path(config, _require_nonempty(pdb, "path", f"training_sets.{name}.pdbs.{pdb_id}"))
            if not path.exists():
                raise ConfigError(f"Training PDB does not exist: {path}")

    for name, testing in testing_sets.items():
        if not isinstance(testing, dict):
            raise ConfigError(f"Testing set '{name}' must be a mapping.")
        merged = deep_merge(defaults, testing)
        merged_without_dna = copy.deepcopy(merged)
        merged_without_dna.pop("dna_mode", None)
        _validate_default_settings(merged_without_dna, f"testing_sets.{name}")
        _validate_dna_mode(testing.get("dna_mode", "ds"), f"testing_sets.{name}.dna_mode")
        pdb = testing.get("pdb")
        if not isinstance(pdb, dict):
            raise ConfigError(f"Testing set '{name}' must define a 'pdb' mapping.")
        pdb_id = _require_nonempty(pdb, "id", f"testing_sets.{name}.pdb")
        pdb_path = resolve_path(config, _require_nonempty(pdb, "path", f"testing_sets.{name}.pdb.{pdb_id}"))
        if not pdb_path.exists():
            raise ConfigError(f"Testing PDB does not exist: {pdb_path}")
        seq_path = resolve_path(config, _require_nonempty(testing, "sequences", f"testing_sets.{name}"))
        if not seq_path.exists():
            raise ConfigError(f"Testing sequence file does not exist: {seq_path}")
        _validate_sequence_file(seq_path)

    jobs = config.get("jobs", [])
    if not isinstance(jobs, list):
        raise ConfigError("'jobs' must be a list.")
    seen_job_names: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ConfigError("Every job must be a mapping.")
        name = _require_nonempty(job, "name", "jobs")
        if name in seen_job_names:
            raise ConfigError(f"Duplicate job name: {name}")
        seen_job_names.add(name)
        training_name = _require_nonempty(job, "training", f"jobs.{name}")
        testing_name = _require_nonempty(job, "testing", f"jobs.{name}")
        if training_name not in training_sets:
            raise ConfigError(f"Job '{name}' references unknown training set '{training_name}'.")
        if testing_name not in testing_sets:
            raise ConfigError(f"Job '{name}' references unknown testing set '{testing_name}'.")


def _normalize_simple_training(config: dict[str, Any], training: Any) -> dict[str, Any]:
    if not isinstance(training, dict):
        raise ConfigError("'training' must be a mapping.")
    pdbs = training.get("pdbs")
    if not isinstance(pdbs, list) or not pdbs:
        raise ConfigError("'training.pdbs' must be a non-empty list.")
    normalized = {key: copy.deepcopy(value) for key, value in training.items() if key != "pdbs"}
    normalized_pdbs = []
    for item in pdbs:
        if isinstance(item, str):
            pdb_id = item
            pdb = {"id": pdb_id, "path": infer_path(config, "training_pdb", pdb_id)}
        elif isinstance(item, dict):
            pdb_id = _require_nonempty(item, "id", "training.pdbs")
            pdb = copy.deepcopy(item)
            pdb.setdefault("path", infer_path(config, "training_pdb", pdb_id))
        else:
            raise ConfigError("Each training.pdbs entry must be a string id or mapping.")
        normalized_pdbs.append(pdb)
    normalized["pdbs"] = normalized_pdbs
    return normalized


def _normalize_simple_testing(config: dict[str, Any], testing: Any) -> dict[str, Any]:
    if not isinstance(testing, dict):
        raise ConfigError("Each testing entry must be a mapping.")
    item = copy.deepcopy(testing)
    if "pdb" not in item:
        pdb_id = _require_nonempty(item, "id", "testing")
        item["pdb"] = {"id": pdb_id, "path": item.pop("path", infer_path(config, "testing_pdb", pdb_id))}
    else:
        pdb_id = _require_nonempty(item["pdb"], "id", "testing.pdb")
        item["pdb"].setdefault("path", infer_path(config, "testing_pdb", pdb_id))
    item.setdefault("sequences", infer_path(config, "testing_sequences", item["pdb"]["id"]))
    item.setdefault("dna_mode", "ds")
    item.pop("id", None)
    item.pop("path", None)
    return item


def _validate_default_settings(settings: dict[str, Any], label: str) -> None:
    if "dna_mode" in settings:
        raise ConfigError(f"{label}.dna_mode is not supported; set dna_mode under testing entries only.")
    phi = settings.get("phi")
    if not isinstance(phi, dict) or not phi.get("name") or not isinstance(phi.get("params"), list):
        raise ConfigError(f"{label}.phi must define 'name' and list 'params'.")
    if len(phi["params"]) != 4:
        raise ConfigError(f"{label}.phi.params must contain 4 values for phi_pairwise_contact_well.")
    if float(settings.get("contact_cutoff_nm", 0)) <= 0:
        raise ConfigError(f"{label}.contact_cutoff_nm must be positive.")
    if int(settings.get("gamma_cutoff_mode", 0)) <= 0:
        raise ConfigError(f"{label}.gamma_cutoff_mode must be positive.")


def _validate_dna_mode(dna_mode: str, label: str) -> None:
    if dna_mode not in SUPPORTED_DNA_MODES:
        raise ConfigError(f"{label} must be one of {sorted(SUPPORTED_DNA_MODES)}.")


def _validate_sequence_file(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip().upper() for line in handle if line.strip()]
    if not lines:
        raise ConfigError(f"Sequence file is empty: {path}")
    invalid = [line for line in lines if set(line) - set("ACGT")]
    if invalid:
        raise ConfigError(f"Sequence file contains non-ACGT sequence: {path}")


def _require_nonempty(mapping: Any, key: str, label: str) -> str:
    if not isinstance(mapping, dict) or key not in mapping or mapping[key] in (None, ""):
        raise ConfigError(f"Missing required '{key}' in {label}.")
    return str(mapping[key])


def _auto_training_name(pdbs: list[dict[str, Any]]) -> str:
    return _safe_name("train_" + "_".join(str(pdb["id"]) for pdb in pdbs))


def _auto_testing_name(testing: dict[str, Any]) -> str:
    return _safe_name("test_" + str(testing["pdb"]["id"]))


def _unique_name(name: str, existing: dict[str, Any]) -> str:
    if name not in existing:
        return name
    index = 2
    while f"{name}_{index}" in existing:
        index += 1
    return f"{name}_{index}"


def _safe_name(name: str) -> str:
    return "_".join(str(name).replace("/", "_").split())
