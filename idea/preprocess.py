"""Raw PDB preprocessing for IDEA protein-DNA inputs.

This ports the useful behavior from the earlier 1_ssdna.py helper into a reusable
module. It strips common solvent/ions/ligands and hydrogens, merges all
protein chains into chain A, and writes DNA as chain B for ssDNA or chains B/C
for dsDNA.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


POSSIBLE_EXTS = (".pdb", ".pdb1", ".ent")
AA_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
DNA_RES = {"DA", "DT", "DG", "DC", "A", "T", "G", "C", "ADE", "THY", "GUA", "CYT", "URI"}
STRIP_RES = {
    "HOH", "WAT", "SO4", "PO4", "CL", "NA", "K", "MG", "MN", "ZN", "CA", "CD", "CO", "CU",
    "SAM", "SAH", "ADO", "MTA", "GOL", "PEG", "MPD", "IPA", "EOH", "ACE", "CIT", "TAR", "FMT", "NO3", "SCN",
}
DNA_MODES = {"auto", "ss", "ds"}


@dataclass(frozen=True)
class PreparedPdb:
    source: Path
    output_id: str
    training_path: Path | None
    testing_path: Path | None
    kept: dict[str, int]


class PreprocessError(ValueError):
    """Raised when raw PDB preprocessing cannot proceed."""


def prepare_data_dir(
    data_dir: str | Path,
    repo_root: str | Path,
    targets: Sequence[str] = ("training", "testing"),
    ids: Sequence[str] | None = None,
    overwrite: bool = True,
    dna_mode: str = "auto",
) -> list[PreparedPdb]:
    data_dir = Path(data_dir).resolve()
    repo_root = Path(repo_root).resolve()
    if not data_dir.exists():
        raise PreprocessError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise PreprocessError(f"Data path is not a directory: {data_dir}")

    dna_mode = _validate_dna_mode(dna_mode)
    target_set = _validate_targets(targets)
    sources = _discover_sources(data_dir, ids)
    if not sources:
        raise PreprocessError(f"No raw PDB files found in {data_dir}")

    prepared: list[PreparedPdb] = []
    for output_id, source in sources:
        prepared.append(prepare_one_pdb(source, output_id, repo_root, target_set, overwrite=overwrite, dna_mode=dna_mode))

    manifest = {
        "data_dir": str(data_dir),
        "targets": sorted(target_set),
        "dna_mode": dna_mode,
        "prepared": [
            {
                "source": str(item.source),
                "id": item.output_id,
                "training_path": str(item.training_path) if item.training_path else None,
                "testing_path": str(item.testing_path) if item.testing_path else None,
                "kept": item.kept,
            }
            for item in prepared
        ],
    }
    out_dir = repo_root / "runs" / "preprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "last_prepare_data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return prepared


def prepare_data_from_config(config: dict, repo_root: str | Path) -> list[PreparedPdb]:
    data = config.get("data")
    if not data:
        return []
    if not isinstance(data, dict):
        raise PreprocessError("config 'data' must be a mapping when provided.")
    config_dir = Path(config.get("_config_dir", ".")).resolve()
    raw_dir = Path(data.get("raw_dir", "data"))
    if not raw_dir.is_absolute():
        raw_dir = (config_dir / raw_dir).resolve()
        if not raw_dir.exists():
            repo_candidate = Path(repo_root).resolve() / data.get("raw_dir", "data")
            raw_dir = repo_candidate.resolve()
    targets = data.get("targets", ["training", "testing"])
    ids = data.get("ids")
    overwrite = bool(data.get("overwrite", True))
    dna_mode = data.get("dna_mode", data.get("structure_mode", "auto"))
    return prepare_data_dir(raw_dir, repo_root, targets=targets, ids=ids, overwrite=overwrite, dna_mode=dna_mode)


def prepare_one_pdb(
    source: str | Path,
    output_id: str,
    repo_root: str | Path,
    targets: Iterable[str] = ("training", "testing"),
    overwrite: bool = True,
    dna_mode: str = "auto",
) -> PreparedPdb:
    source = Path(source).resolve()
    repo_root = Path(repo_root).resolve()
    if not source.exists():
        raise PreprocessError(f"Raw PDB file does not exist: {source}")
    target_set = _validate_targets(targets)
    dna_mode = _validate_dna_mode(dna_mode)

    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    rewritten, kept = rewrite_pdb_by_resclass(lines, dna_mode=dna_mode)
    if kept.get("A", 0) == 0:
        raise PreprocessError(f"No protein residues were retained from {source}")
    if kept.get("B", 0) == 0:
        raise PreprocessError(f"No DNA residues were retained from {source}")
    if dna_mode == "ds" and kept.get("C", 0) == 0:
        raise PreprocessError(f"dna_mode=ds requires two DNA chains, but only one was retained from {source}")

    training_path = None
    testing_path = None
    if "training" in target_set:
        training_path = repo_root / "training" / "PDBs" / f"{output_id}_modified.pdb"
        _write_output(training_path, rewritten, overwrite=overwrite)
    if "testing" in target_set:
        testing_path = repo_root / "testing" / "PDBs" / f"{output_id}_modified.pdb"
        _write_output(testing_path, rewritten, overwrite=overwrite)

    return PreparedPdb(source=source, output_id=output_id, training_path=training_path, testing_path=testing_path, kept=dict(kept))


def rewrite_pdb_by_resclass(lines: Sequence[str], dna_mode: str = "auto") -> tuple[list[str], Counter]:
    """Rewrite one protein-DNA structure into IDEA's expected chain convention.

    IDEA's current phi code uses one protein chain (`A`), so all protein chains
    are merged into chain `A`. Because different protein chains often reuse the
    same residue numbers, protein residues are renumbered sequentially while atom
    coordinates are left unchanged. DNA handling depends on `dna_mode`: `ss`
    keeps the largest DNA chain as `B`; `ds` keeps the two largest DNA chains as
    `B` and `C`; `auto` chooses `ds` when two or more DNA chains are present,
    otherwise `ss`.
    """
    dna_mode = _validate_dna_mode(dna_mode)
    chain_counts = _collect_chain_type_counts(lines)
    protein_chains = _chains_by_count(chain_counts, "protein")
    dna_chains = _chains_by_count(chain_counts, "dna")
    if not protein_chains:
        raise PreprocessError("No protein chain was found in the raw PDB.")
    if not dna_chains:
        raise PreprocessError("No DNA chain was found in the raw PDB.")

    resolved_dna_mode = "ds" if dna_mode == "auto" and len(dna_chains) >= 2 else "ss" if dna_mode == "auto" else dna_mode
    if resolved_dna_mode == "ds" and len(dna_chains) < 2:
        raise PreprocessError("dna_mode=ds requires at least two DNA chains in the raw PDB.")

    protein_chain_set = set(protein_chains)
    dna_chain_map = {dna_chains[0]: "B"}
    if resolved_dna_mode == "ds":
        dna_chain_map[dna_chains[1]] = "C"

    out: list[str] = []
    kept: Counter = Counter()
    seen_atoms: set[tuple[str, str, str, str]] = set()
    protein_residue_numbers: dict[tuple[str, str, str], int] = {}
    next_protein_resseq = 1

    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        resn = line[17:20].strip().upper()
        atom = line[12:16]
        if resn in STRIP_RES or _is_h(atom):
            continue

        source_chain = _line_chain(line)
        chars = list(line)
        if len(chars) < 80:
            chars += [" "] * (80 - len(chars))

        if resn in AA_RES:
            if source_chain not in protein_chain_set:
                continue
            new_chain = "A"
            residue_key = _residue_key(line)
            if residue_key not in protein_residue_numbers:
                protein_residue_numbers[residue_key] = next_protein_resseq
                next_protein_resseq += 1
            chars[22:26] = f"{protein_residue_numbers[residue_key]:4d}"[-4:]
            chars[26] = " "
        elif resn in DNA_RES:
            new_chain = dna_chain_map.get(source_chain)
            if new_chain is None:
                continue
        else:
            continue

        chars[21] = new_chain
        rewritten = "".join(chars)
        atom_key = (new_chain, rewritten[22:26], rewritten[26:27], rewritten[12:16].strip())
        if atom_key in seen_atoms:
            continue
        seen_atoms.add(atom_key)
        out.append(rewritten)
        kept[new_chain] += 1
    return out, kept


def _collect_chain_type_counts(lines: Sequence[str]) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        resn = line[17:20].strip().upper()
        if resn in STRIP_RES or _is_h(line[12:16]):
            continue
        chain = _line_chain(line)
        if resn in AA_RES:
            counts[chain]["protein"] += 1
        elif resn in DNA_RES:
            counts[chain]["dna"] += 1
    return counts


def _chains_by_count(chain_counts: dict[str, Counter], resclass: str) -> list[str]:
    return [
        chain
        for chain, _count in sorted(
            ((chain, counts[resclass]) for chain, counts in chain_counts.items() if counts[resclass] > 0),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _discover_sources(data_dir: Path, ids: Sequence[str] | None) -> list[tuple[str, Path]]:
    if ids:
        sources = []
        for raw_id in ids:
            if not isinstance(raw_id, str):
                raise PreprocessError("data.ids entries must be strings.")
            source = _find_source(data_dir, raw_id)
            sources.append((raw_id, source))
        return sources

    discovered = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in POSSIBLE_EXTS:
            continue
        if path.stem.endswith("_modified"):
            continue
        discovered.append((path.stem, path))
    return discovered


def _find_source(data_dir: Path, raw_id: str) -> Path:
    candidates = [data_dir / raw_id]
    if Path(raw_id).suffix == "":
        candidates.extend(data_dir / f"{raw_id}{ext}" for ext in POSSIBLE_EXTS)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise PreprocessError(f"Could not find raw PDB for id '{raw_id}'. Tried: {tried}")


def _validate_targets(targets: Iterable[str]) -> set[str]:
    target_set = {str(target) for target in targets}
    allowed = {"training", "testing"}
    unknown = target_set - allowed
    if unknown:
        raise PreprocessError(f"Unknown prepare-data target(s): {sorted(unknown)}. Allowed: {sorted(allowed)}")
    if not target_set:
        raise PreprocessError("At least one prepare-data target is required.")
    return target_set


def _validate_dna_mode(dna_mode: str) -> str:
    mode = str(dna_mode).strip().lower()
    if mode not in DNA_MODES:
        raise PreprocessError(f"Unknown prepare-data dna_mode '{dna_mode}'. Allowed: {sorted(DNA_MODES)}")
    return mode


def _write_output(path: Path, lines: Sequence[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise PreprocessError(f"Output already exists and overwrite is false: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _residue_key(line: str) -> tuple[str, str, str]:
    return (_line_chain(line), line[22:26], line[26:27])


def _line_chain(line: str) -> str:
    return line[21] if len(line) > 21 and line[21].strip() else " "


def _is_h(atom_name: str) -> bool:
    return atom_name.strip().upper().startswith("H")
