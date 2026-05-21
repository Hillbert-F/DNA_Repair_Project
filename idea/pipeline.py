"""Pipeline orchestration and artifact caching for IDEA."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .config import (
    ConfigError,
    deep_merge,
    phi_params_string,
    resolve_path,
)
from .energy import ENERGY_CALCULATION_VERSION, calculate_energy_files
from .hashing import file_sha256, stable_hash


PIPELINE_VERSION = "idea-pipeline-v1"


@dataclass(frozen=True)
class Artifact:
    kind: str
    key: str
    path: Path
    manifest_path: Path
    reused: bool = False


class PipelineError(RuntimeError):
    """Raised when an IDEA pipeline step fails."""


class IdeaPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        repo_root: str | Path | None = None,
        runs_root: str | Path | None = None,
    ):
        self.config = config
        self.repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[1]
        self.runs_root = Path(runs_root).resolve() if runs_root else self.repo_root / "runs"
        self.cache_root = self.runs_root / "cache"
        self.experiments_root = self.runs_root / "experiments"

    def dry_run(self) -> dict[str, Any]:
        training = {
            name: self.training_key(name, spec)
            for name, spec in self.config.get("training_sets", {}).items()
        }
        testing = {
            name: self.testing_key(name, spec)
            for name, spec in self.config.get("testing_sets", {}).items()
        }
        jobs = []
        for job in self.config.get("jobs", []):
            training_key = training[job["training"]]
            testing_key = testing[job["testing"]]
            jobs.append(
                {
                    "name": job["name"],
                    "training": job["training"],
                    "testing": job["testing"],
                    "training_hash": training_key,
                    "testing_hash": testing_key,
                    "energy_hash": self.energy_key_for_artifacts(training_key, testing_key),
                    "training_artifact": str(self._cache_dir("training", training_key, job["training"], migrate=False).relative_to(self.repo_root)),
                    "testing_artifact": str(self._cache_dir("testing", testing_key, job["testing"], migrate=False).relative_to(self.repo_root)),
                    "energy_artifact": str(self._cache_dir("energy", self.energy_key_for_artifacts(training_key, testing_key), f"{job['training']}__{job['testing']}", migrate=False).relative_to(self.repo_root)),
                }
            )
        return {"training": training, "testing": testing, "jobs": jobs}

    def train(self, training_name: str, force: bool = False) -> Artifact:
        try:
            spec = self.config["training_sets"][training_name]
        except KeyError as exc:
            raise ConfigError(f"Unknown training set: {training_name}") from exc

        key = self.training_key(training_name, spec)
        artifact_dir = self._cache_dir("training", key, training_name)
        gamma_path = artifact_dir / "gamma_filtered.txt"
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists() and gamma_path.exists() and not force:
            self._info(f"Training cache hit: {training_name} -> {artifact_dir.relative_to(self.repo_root)}")
            return Artifact("training", key, artifact_dir, manifest_path, reused=True)

        self._info(f"Training cache miss: {training_name} -> {artifact_dir.relative_to(self.repo_root)}")
        if force and artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = artifact_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        work_dir = artifact_dir / "work"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        shutil.copytree(self.repo_root / "training", work_dir / "training", ignore=_ignore_runtime_files)

        stage = work_dir / "training"
        resolved = self._resolved_training_spec(training_name, spec)
        self._info(f"Preparing staged training workspace for {training_name}")
        self._prepare_training_stage(stage, resolved)
        self._info(f"Running IDEA training for {training_name}")
        self._run_command(["bash", "train.sh"], cwd=stage, log_path=logs_dir / "train.log")
        gamma_stage = stage / "optimization" / "for_training_gamma"
        params = phi_params_string(self.config, spec)
        source_prefix = (
            stage
            / "optimization"
            / "for_training_gamma"
            / "gammas"
            / "randomized_decoy"
            / f"native_trainSetFiles_phi_pairwise_contact_well{params}_gamma_filtered"
        )
        if not source_prefix.exists():
            raise PipelineError(
                "Expected gamma file was not generated after training. "
                "Check logs/train.log for the real failure. On Colab this often means "
                "optimize_gamma.py was killed because the session ran out of RAM; try fewer "
                "training decoys or a high-RAM/runtime-local environment. "
                f"Missing file: {source_prefix}"
            )
        visualize_status = "created"
        self._info(f"Generating training visualizations for {training_name}")
        try:
            self._run_command(["python", "visualize.py"], cwd=gamma_stage, log_path=logs_dir / "visualize.log")
        except PipelineError as exc:
            visualize_status = "failed"
            self._info(
                f"Training visualization failed for {training_name}; continuing with generated gamma. {exc}"
            )
        shutil.copy2(source_prefix, gamma_path)
        _copy_optional_dir(stage / "optimization" / "for_training_gamma" / "phis", artifact_dir / "phis")
        _copy_optional_dir(stage / "optimization" / "for_training_gamma" / "tms", artifact_dir / "tms")
        _copy_optional_dir(stage / "optimization" / "for_training_gamma" / "visualize", artifact_dir / "visualize")

        manifest = {
            "kind": "training",
            "pipeline_version": PIPELINE_VERSION,
            "idea_version": __version__,
            "hash": key,
            "label": training_name,
            "training_set": training_name,
            "resolved": resolved,
            "gamma_filtered": str(gamma_path.relative_to(self.repo_root)),
            "visualize": str((artifact_dir / "visualize").relative_to(self.repo_root)),
            "visualize_status": visualize_status,
        }
        _write_json(manifest_path, manifest)
        return Artifact("training", key, artifact_dir, manifest_path, reused=False)

    def test(self, testing_name: str, force: bool = False) -> Artifact:
        try:
            spec = self.config["testing_sets"][testing_name]
        except KeyError as exc:
            raise ConfigError(f"Unknown testing set: {testing_name}") from exc

        key = self.testing_key(testing_name, spec)
        artifact_dir = self._cache_dir("testing", key, testing_name)
        phi_decoys = artifact_dir / "phi_decoys.txt"
        phi_native = artifact_dir / "phi_native.txt"
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists() and phi_decoys.exists() and phi_native.exists() and not force:
            self._info(f"Testing cache hit: {testing_name} -> {artifact_dir.relative_to(self.repo_root)}")
            return Artifact("testing", key, artifact_dir, manifest_path, reused=True)

        self._info(f"Testing cache miss: {testing_name} -> {artifact_dir.relative_to(self.repo_root)}")
        if force and artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = artifact_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        work_dir = artifact_dir / "work"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        shutil.copytree(self.repo_root / "testing", work_dir / "testing", ignore=_ignore_runtime_files)

        stage = work_dir / "testing"
        resolved = self._resolved_testing_spec(testing_name, spec)
        self._info(f"Preparing staged testing workspace for {testing_name}")
        self._prepare_testing_stage(stage, resolved)
        self._info(f"Running IDEA testing phi generation for {testing_name}")
        self._run_command(["bash", "run_test_cli.sh", resolved["pdb"]["id"]], cwd=stage, log_path=logs_dir / "test.log")

        params = phi_params_string(self.config, spec)
        source_decoys = stage / "phis" / f"phi_pairwise_contact_well_native_decoys_CPLEX_randomization_{params}"
        source_native = stage / "phis" / f"phi_pairwise_contact_well_native_native_{params}"
        if not source_decoys.exists():
            raise PipelineError(f"Expected testing phi file was not generated: {source_decoys}")
        if not source_native.exists():
            raise PipelineError(f"Expected native phi file was not generated: {source_native}")
        shutil.copy2(source_decoys, phi_decoys)
        shutil.copy2(source_native, phi_native)

        manifest = {
            "kind": "testing",
            "pipeline_version": PIPELINE_VERSION,
            "idea_version": __version__,
            "hash": key,
            "label": testing_name,
            "testing_set": testing_name,
            "resolved": resolved,
            "phi_decoys": str(phi_decoys.relative_to(self.repo_root)),
            "phi_native": str(phi_native.relative_to(self.repo_root)),
        }
        _write_json(manifest_path, manifest)
        return Artifact("testing", key, artifact_dir, manifest_path, reused=False)

    def energy_from_artifacts(self, training_artifact: Artifact, testing_artifact: Artifact, force: bool = False) -> Artifact:
        key = self.energy_key_for_artifacts(training_artifact.key, testing_artifact.key)
        energy_label = f"{_artifact_label(training_artifact)}__{_artifact_label(testing_artifact)}"
        artifact_dir = self._cache_dir("energy", key, energy_label)
        energy_path = artifact_dir / "Energy_mg.txt"
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists() and energy_path.exists() and not force:
            self._info(f"Energy cache hit: {_artifact_label(training_artifact)} + {_artifact_label(testing_artifact)} -> {artifact_dir.relative_to(self.repo_root)}")
            return Artifact("energy", key, artifact_dir, manifest_path, reused=True)
        self._info(f"Energy cache miss: {_artifact_label(training_artifact)} + {_artifact_label(testing_artifact)} -> {artifact_dir.relative_to(self.repo_root)}")
        if force and artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        calculate_energy_files(
            training_artifact.path / "gamma_filtered.txt",
            testing_artifact.path / "phi_decoys.txt",
            artifact_dir,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "kind": "energy",
                "hash": key,
                "label": energy_label,
                "training_hash": training_artifact.key,
                "testing_hash": testing_artifact.key,
            }
        )
        _write_json(manifest_path, manifest)
        return Artifact("energy", key, artifact_dir, manifest_path, reused=False)

    def run_jobs(self, force: bool = False) -> Path:
        experiment_dir = self.experiments_root / str(self.config["run_id"])
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with (experiment_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_public_config(self.config), handle, sort_keys=False)

        rows = []
        jobs = self.config.get("jobs", [])
        self._info(f"Starting IDEA run '{self.config['run_id']}' with {len(jobs)} job(s)")
        for index, job in enumerate(jobs, start=1):
            self._info(f"Job {index}/{len(jobs)}: {job['name']}")
            training_artifact = self.train(job["training"], force=force)
            testing_artifact = self.test(job["testing"], force=force)
            energy_artifact = self.energy_from_artifacts(training_artifact, testing_artifact, force=force)

            job_dir = experiment_dir / _safe_name(job["name"])
            job_dir.mkdir(exist_ok=True)
            (job_dir / "training_artifact.txt").write_text(str(training_artifact.path) + "\n", encoding="utf-8")
            (job_dir / "testing_artifact.txt").write_text(str(testing_artifact.path) + "\n", encoding="utf-8")
            (job_dir / "energy_artifact.txt").write_text(str(energy_artifact.path) + "\n", encoding="utf-8")
            rows.append(
                {
                    "job": job["name"],
                    "training": job["training"],
                    "testing": job["testing"],
                    "training_hash": training_artifact.key,
                    "testing_hash": testing_artifact.key,
                    "energy_hash": energy_artifact.key,
                    "training_reused": training_artifact.reused,
                    "testing_reused": testing_artifact.reused,
                    "energy_reused": energy_artifact.reused,
                    "training_artifact_path": training_artifact.path,
                    "testing_artifact_path": testing_artifact.path,
                    "energy_artifact_path": energy_artifact.path,
                    "energy_path": energy_artifact.path / "Energy_mg.txt",
                }
            )

        with (experiment_dir / "jobs_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "job",
                "training",
                "testing",
                "training_hash",
                "testing_hash",
                "energy_hash",
                "training_reused",
                "testing_reused",
                "energy_reused",
                "training_artifact_path",
                "testing_artifact_path",
                "energy_artifact_path",
                "energy_path",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return experiment_dir

    def training_key(self, training_name: str, spec: dict[str, Any]) -> str:
        resolved = self._resolved_training_spec(training_name, spec)
        return stable_hash({"kind": "training", "version": PIPELINE_VERSION, "resolved": resolved})

    def testing_key(self, testing_name: str, spec: dict[str, Any]) -> str:
        resolved = self._resolved_testing_spec(testing_name, spec)
        return stable_hash({"kind": "testing", "version": PIPELINE_VERSION, "resolved": resolved})

    def energy_key_for_artifacts(self, training_key: str, testing_key: str) -> str:
        return stable_hash(
            {
                "kind": "energy",
                "version": ENERGY_CALCULATION_VERSION,
                "training_hash": training_key,
                "testing_hash": testing_key,
            }
        )

    def _cache_dir(self, kind: str, key: str, label: str, migrate: bool = True) -> Path:
        kind_dir = self.cache_root / kind
        readable = _safe_name(label)
        new_dir = kind_dir / f"{readable}__{key}"
        old_hash_dir = kind_dir / key
        if old_hash_dir.exists() and not new_dir.exists():
            if migrate:
                kind_dir.mkdir(parents=True, exist_ok=True)
                old_hash_dir.rename(new_dir)
            else:
                return new_dir
        elif not new_dir.exists():
            matches = sorted(kind_dir.glob(f"*__{key}")) if kind_dir.exists() else []
            if matches:
                return matches[0]
        return new_dir

    def _info(self, message: str) -> None:
        print(f"[IDEA] {message}", flush=True)

    def _resolved_training_spec(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        merged = deep_merge(self.config["defaults"], spec)
        pdbs = []
        for pdb in spec["pdbs"]:
            path = resolve_path(self.config, pdb["path"])
            pdbs.append({"id": str(pdb["id"]), "path": str(path), "sha256": file_sha256(path)})
        return {
            "name": name,
            "pdbs": pdbs,
            "protein_chain": str(merged["protein_chain"]),
            "contact_cutoff_nm": float(merged["contact_cutoff_nm"]),
            "gamma_cutoff_mode": int(merged["gamma_cutoff_mode"]),
            "phi": merged["phi"],
            "training_decoys": {
                "dna": int(merged["training_decoys"]["dna"]),
                "protein": int(merged["training_decoys"]["protein"]),
            },
        }

    def _resolved_testing_spec(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        merged = deep_merge(self.config["defaults"], spec)
        pdb_path = resolve_path(self.config, spec["pdb"]["path"])
        seq_path = resolve_path(self.config, spec["sequences"])
        return {
            "name": name,
            "pdb": {"id": str(spec["pdb"]["id"]), "path": str(pdb_path), "sha256": file_sha256(pdb_path)},
            "sequences": {"path": str(seq_path), "sha256": file_sha256(seq_path)},
            "dna_mode": str(merged["dna_mode"]),
            "protein_chain": str(merged["protein_chain"]),
            "contact_cutoff_nm": float(merged["contact_cutoff_nm"]),
            "phi": merged["phi"],
            "max_testing_decoys": int(merged["max_testing_decoys"]),
        }

    def _prepare_training_stage(self, stage: Path, resolved: dict[str, Any]) -> None:
        _write_phi_list(stage / "phi1_list.txt", resolved["phi"])
        _write_lines(stage / "proteinList.txt", [pdb["id"] for pdb in resolved["pdbs"]])
        for directory in [
            stage / "optimization" / "for_training_gamma" / "native_structures_pdbs_with_virtual_cbs",
            stage / "optimization" / "for_training_gamma" / "phis",
            stage / "optimization" / "for_training_gamma" / "tms",
            stage / "optimization" / "for_training_gamma" / "gammas" / "randomized_decoy",
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        pdb_dir = stage / "PDBs"
        shutil.rmtree(pdb_dir, ignore_errors=True)
        pdb_dir.mkdir(parents=True)
        for pdb in resolved["pdbs"]:
            shutil.copy2(pdb["path"], pdb_dir / f"{pdb['id']}_modified.pdb")
        _patch_text(stage / "optimization" / "for_bindingE" / "template" / "cmd.preprocessing.sh", {
            r"export cutoff=[^\n]+": f"export cutoff={resolved['contact_cutoff_nm']}",
        })
        _patch_text(stage / "optimization" / "for_bindingE" / "template" / "sequences" / "generate_decoy_seq_DNA.py", {
            r"num_decoys=\[[0-9]+\]": f"num_decoys=[{resolved['training_decoys']['dna']}]",
        })
        _patch_text(stage / "optimization" / "for_bindingE" / "template" / "sequences" / "generate_decoy_seq_prot.py", {
            r"num_decoys=[0-9]+": f"num_decoys={resolved['training_decoys']['protein']}",
        })
        _patch_text(stage / "optimization" / "for_training_gamma" / "optimize_gamma.py", {
            r"cutoff_mode = [0-9]+": f"cutoff_mode = {resolved['gamma_cutoff_mode']}",
        })

    def _prepare_testing_stage(self, stage: Path, resolved: dict[str, Any]) -> None:
        _write_phi_list(stage / "phi1_list.txt", resolved["phi"])
        pdb_dir = stage / "PDBs"
        shutil.rmtree(pdb_dir, ignore_errors=True)
        pdb_dir.mkdir(parents=True)
        shutil.copy2(resolved["pdb"]["path"], pdb_dir / f"{resolved['pdb']['id']}_modified.pdb")
        seq_dir = stage / "sequences"
        shutil.copy2(resolved["sequences"]["path"], seq_dir / "input_sequences.seq")
        shutil.copy2(resolved["sequences"]["path"], seq_dir / "dna_half.seq")
        self._validate_sequence_lengths_for_testing(resolved)
        _write_testing_driver(stage / "run_test_cli.sh", resolved)
        os.chmod(stage / "run_test_cli.sh", 0o755)

    def _validate_sequence_lengths_for_testing(self, resolved: dict[str, Any]) -> None:
        dna_count = count_dna_residues(Path(resolved["pdb"]["path"]))
        seqs = read_sequence_lines(Path(resolved["sequences"]["path"]))
        if resolved["dna_mode"] == "ss":
            expected = dna_count
        else:
            if dna_count % 2:
                raise ConfigError(
                    f"Testing PDB {resolved['pdb']['path']} has {dna_count} DNA residues; "
                    "ds mode expects an even total DNA length."
                )
            expected = dna_count // 2
        bad = [seq for seq in seqs if len(seq) != expected]
        if bad:
            raise ConfigError(
                f"Sequence length mismatch for testing set '{resolved['name']}': "
                f"dna_mode={resolved['dna_mode']} expects length {expected}, "
                f"but found length {len(bad[0])}."
            )

    def _run_command(self, cmd: list[str], cwd: Path, log_path: Path) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        display_cwd = _run_relative(cwd, self.repo_root)
        self._info(f"$ {' '.join(cmd)}  (cwd: {display_cwd})")
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("$ " + " ".join(cmd) + f"\nCWD: {cwd}\n\n")
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
            return_code = process.wait()
        if return_code != 0:
            raise PipelineError(f"Command failed with exit code {return_code}. See log: {log_path}")


def _artifact_label(artifact: Artifact) -> str:
    name = artifact.path.name
    suffix = f"__{artifact.key}"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def read_sequence_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip().upper() for line in handle if line.strip()]


def count_dna_residues(pdb_path: Path) -> int:
    dna_res = {"DA", "DT", "DG", "DC", "A", "T", "G", "C", "ADE", "THY", "GUA", "CYT"}
    seen = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resn = line[17:20].strip().upper()
            if resn not in dna_res:
                continue
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip() if len(line) > 26 else ""
            seen.add((chain, resseq, icode))
    if not seen:
        raise ConfigError(f"No DNA residues found in testing PDB: {pdb_path}")
    return len(seen)


def _write_testing_driver(path: Path, resolved: dict[str, Any]) -> None:
    dna_setup = ""
    if resolved["dna_mode"] == "ss":
        dna_setup = "cp input_sequences.seq dna.seq\n"
    else:
        dna_setup = "\n".join(
            [
                "python reverse_complement.py dna_half.seq dna_half_complement.seq",
                "python merge.py",
            ]
        ) + "\n"

    script = f"""#!/bin/bash
set -euo pipefail

export PDBid=$1
protChain="{resolved['protein_chain']}"

find . -type f -name "proteinList.txt" -not -path "./proteinList.txt" -exec cp ./proteinList.txt {{}} \\;
cp PDBs/${{PDBid}}_modified.pdb native_structures_pdbs_with_virtual_cbs/native.pdb

cp proteins_list.txt sequences/
cp native_structures_pdbs_with_virtual_cbs/native.pdb sequences/
cd sequences/
python buildseq.py native
{dna_setup}python mapDNAseq_reverse.py dna.seq dna_modeller.seq
python combine_DNAPro.py
export cutoff={resolved['contact_cutoff_nm']}
python find_cm_residues.py native.pdb $cutoff randomize_position_prot.txt randomize_position_DNA.txt

rm -rf DNA_randomization
mkdir -p DNA_randomization
cp randomize_position_DNA.txt native.seq native.decoys DNA_randomization/

rm -rf CPLEX_randomization
mkdir -p CPLEX_randomization
cat DNA_randomization/native.decoys > CPLEX_randomization/native.decoys

cd ../
grep "CA\\|O5'" native_structures_pdbs_with_virtual_cbs/native.pdb > tmp.txt
tot_resnum=$(grep '^ATOM' tmp.txt | wc -l)
python create_tms.py sequences/DNA_randomization/randomize_position_DNA.txt $tot_resnum $PDBid
sed "s/CPLEX_NAME/$PDBid/g; s/PROT_CHAIN/$protChain/g" template_evaluate_phi.py > evaluate_phi.py
python evaluate_phi.py
"""
    path.write_text(script, encoding="utf-8")


def _write_phi_list(path: Path, phi: dict[str, Any]) -> None:
    path.write_text("{} {}\n".format(phi["name"], " ".join(str(x) for x in phi["params"])), encoding="utf-8")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def _patch_text(path: Path, replacements: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    path.write_text(text, encoding="utf-8")


def _copy_optional_dir(source: Path, dest: Path) -> None:
    if not source.exists():
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def _run_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _ignore_runtime_files(dir_path: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".ipynb_checkpoints"}
    if Path(dir_path).name in {"phis", "tms", "gammas", "visualize", "native_structures_pdbs_with_virtual_cbs"}:
        return set(names)
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "job"
