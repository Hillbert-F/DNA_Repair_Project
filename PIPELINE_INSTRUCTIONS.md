# IDEA Config-Driven Pipeline Instructions

This document describes the refactored IDEA pipeline in `IDEA_Model`. The goal is to run training, testing, and energy calculation without editing `proteinList.txt`, replacing `dna_half.seq`, manually copying gamma/phi files, or overwriting previous outputs.

## 1. What Changed

### New user-facing command

Run IDEA from the repository root with:

```bash
conda activate IDEA
python -m idea.cli run --config configs/input_settings.yaml
```

If `conda` is not already on your `PATH`, use the full conda path on this cluster:

```bash
/home/hfan7/miniconda3/bin/conda run -n IDEA \
  python -m idea.cli run --config configs/input_settings.yaml
```

### New Python package

The new orchestration code lives in:

```text
idea/cli.py       command-line interface
idea/config.py    YAML loading, defaults, validation, auto naming
idea/pipeline.py  artifact hashing, staging, train/test/energy orchestration
idea/energy.py    explicit-path energy calculation
idea/hashing.py   stable content-hash helpers
```

The original IDEA training/testing scripts are still used as implementation pieces, but they now run inside private staged work directories under `runs/cache/.../work/`. Users should no longer edit or run `training/train.sh` and `testing/test.sh` directly for normal use.

## 2. Minimal YAML Format

The simplified config looks like this:

```yaml
run_id: max_example_batch  # optional; auto-generated if omitted

defaults:
  protein_chain: A
  contact_cutoff_nm: 1.2
  gamma_cutoff_mode: 60
  phi:
    name: phi_pairwise_contact_well
    params: [-8.0, 8.0, 0.7, 10]
  training_decoys:
    dna: 1000
    protein: 10000
  max_testing_decoys: 1000000

training:
  pdbs:
    - 1hlo
    - 1nkp
    - 1nlw

testing:
  - id: 1hlo
    dna_mode: ds
```

For built-in repo data, a training PDB id like `1hlo` automatically resolves to:

```text
training/PDBs/1hlo_modified.pdb
```

A testing PDB id like `1hlo` automatically resolves to:

```text
testing/PDBs/1hlo_modified.pdb
```

If `sequences` is omitted, testing sequences default to:

```text
testing/sequences/dna_half.seq
```

For external data, provide paths explicitly:

```yaml
training:
  pdbs:
    - id: EPECDarT
      path: /path/to/EPECDarT_modified.pdb
    - id: ThermusDarT
      path: /path/to/ThermusDarT_modified.pdb

testing:
  - id: EPECDarT
    path: /path/to/EPECDarT_modified.pdb
    sequences: /path/to/dart_sequences.seq
    dna_mode: ss
```

## 3. Field Meanings

### `run_id`

Optional. If you set it, the experiment summary is written to:

```text
runs/experiments/<run_id>/
```

If you omit it, the pipeline auto-generates one from the config filename and current timestamp, for example:

```text
example_20260519_143012
```

This only affects the human-readable experiment folder. Artifact cache keys are based on input contents and settings, not on `run_id`.

### `defaults`

These are global settings and can be modified:

```text
protein_chain
contact_cutoff_nm
gamma_cutoff_mode
phi
training_decoys
max_testing_decoys
```

`dna_mode` is intentionally not allowed under `defaults`. DNA mode belongs to each testing entry because different testing structures may be single-stranded or double-stranded.

### `training`

Normal users only need to list training PDB ids:

```yaml
training:
  pdbs:
    - 1hlo
    - 1nkp
```

Each item can also be a mapping with `id` and `path`:

```yaml
training:
  pdbs:
    - id: custom1
      path: data/custom1_modified.pdb
```

The internal training set name is generated automatically as `train_<ids>`, for example:

```text
train_1hlo_1nkp_1nlw
```

### `testing`

Each testing entry needs an `id` and `dna_mode`; `path` and `sequences` can be explicit or inferred for built-in data.

```yaml
testing:
  - id: 1hlo
    dna_mode: ds
  - id: EPECDarT
    path: data/EPECDarT_modified.pdb
    sequences: data/dart_5mers.seq
    dna_mode: ss
```

DNA modes:

```text
ss  single-stranded DNA; use the provided sequence as the full DNA sequence
ds  double-stranded DNA; use provided sequence as one half, then run reverse_complement.py and merge.py
```

The internal testing set name is generated automatically as `test_<id>`.

### `jobs`

Normal users do not need to write `jobs`. If omitted, IDEA automatically runs every configured training set against every configured testing set.

For the simple config above, the generated job name is:

```text
train_1hlo_1nkp_1nlw__test_1hlo
```

Advanced users can still provide explicit `training_sets`, `testing_sets`, and `jobs`, but the minimal `training` / `testing` format is the recommended interface.

## 3.5 Optional Raw-PDB Preprocessing Head

If your PDB files are still raw/unpolished, put them in a data directory and let IDEA prepare them before the pipeline starts. For example, a raw file:

```text
data/ThermusDarT.pdb
```

can become:

```text
training/PDBs/ThermusDarT_modified.pdb
testing/PDBs/ThermusDarT_modified.pdb
```

The standalone command is:

```bash
python -m idea.cli prepare-data --data-dir data
```

By default, `--dna-mode auto` keeps one DNA chain as `B` for ssDNA and two DNA chains as `B/C` for dsDNA. You can force either mode:

```bash
python -m idea.cli prepare-data --data-dir data --dna-mode ss
python -m idea.cli prepare-data --data-dir data --dna-mode ds
```

You can limit which files and where they are written:

```bash
python -m idea.cli prepare-data --data-dir data --ids ThermusDarT ThermusDarTAF
python -m idea.cli prepare-data --data-dir data --targets training
```

Or enable the same preprocessing as a head step in the YAML. `run`, `train`, and `test` execute this before config validation; `run --dry-run` does not write files.

```yaml
data:
  raw_dir: data
  targets: [training, testing]
  dna_mode: auto  # auto, ss, or ds for raw PDB preprocessing
  ids: [ThermusDarT, ThermusDarTAF]  # optional; omit to process all raw PDBs
  overwrite: true
```

The preprocessing step ports and generalizes the useful behavior from `1_ssdna.py`: it merges all protein chains into chain `A`, writes ssDNA as chain `B`, writes dsDNA as chains `B/C`, renumbers merged protein residues to avoid duplicate residue IDs, strips hydrogens/common solvent/ions/ligands, removes duplicate atoms created by chain normalization, and writes `{id}_modified.pdb` without appending an extra `END` record. After this step, the YAML `training` and `testing` ids should still use the clean stem before `_modified.pdb`, for example `ThermusDarT`.

## 4. How to Run

### Step 1: Dry run

Dry-run validates the config and shows cache keys without running expensive phi/gamma generation:

```bash
conda activate IDEA
python -m idea.cli run --config configs/input_settings.yaml --dry-run
```

Cluster-safe version:

```bash
/home/hfan7/miniconda3/bin/conda run -n IDEA \
  python -m idea.cli run --config configs/input_settings.yaml --dry-run
```

### Step 2: Full pipeline

```bash
conda activate IDEA
python -m idea.cli run --config configs/input_settings.yaml
```

This runs:

```text
training gamma -> training visualization -> testing phi -> energy calculation
```

### Step 3: Force recomputation when needed

By default, matching artifacts are reused. To recompute everything for the configured jobs:

```bash
python -m idea.cli run --config configs/input_settings.yaml --force
```

## 4.5 Progress Output

The CLI streams the underlying `train.sh`, generated testing script, and
`visualize.py` output to the terminal in real time. The same text is also saved
under each artifact's `logs/` directory. Cache hits and misses are printed before
each stage, for example:

```text
[IDEA] Training cache hit: train_ThermusDarTAF -> runs/cache/training/train_ThermusDarTAF__5383d22b457f2f9c
[IDEA] Running IDEA testing phi generation for test_ThermusDarTAF
```

## 5. Output Layout

Artifacts are stored by content hash with readable prefixes:

```text
runs/cache/training/<training_name>__<training_hash>/
  gamma_filtered.txt
  manifest.json
  phis/
  tms/
  visualize/
    decoy_phi.pdf
    native_phi.pdf
    trained_gamma.pdf
  logs/
    train.log
    visualize.log

runs/cache/testing/<testing_name>__<testing_hash>/
  phi_decoys.txt
  phi_native.txt
  manifest.json
  logs/test.log

runs/cache/energy/<training_name>__<testing_name>__<energy_hash>/
  Energy_mg.txt
  manifest.json

runs/experiments/<run_id>/
  resolved_config.yaml
  jobs_summary.csv
  <auto_job_name>/
    training_artifact.txt
    testing_artifact.txt
    energy_artifact.txt
```

The hash suffix is still the stable cache identity. The readable prefix is there so
you can scan a batch run in the file browser without opening every manifest. Old
hash-only cache directories are automatically renamed to the readable form the
next time that artifact is reused by a real `run`, `train`, or `test` command.

This solves the original overwrite problems:

- Same training + different testing: gamma cache is reused; phi outputs differ.
- Different training + same testing: phi cache is reused; gamma outputs differ.
- Different gamma/phi pairs: energy outputs go to different energy hashes.

## 6. What Happened to `train.sh` and `test.sh`?

### Training

`training/train.sh` is not moved into `idea/pipeline.py`. Instead, the pipeline copies the whole `training/` folder into:

```text
runs/cache/training/<training_name>__<training_hash>/work/training/
```

Then it writes the staged `proteinList.txt`, copies the configured PDBs into staged `PDBs/`, patches run-specific decoy counts, contact cutoff, and gamma cutoff, and runs staged:

```bash
bash train.sh
```

So the original training script still performs the core training, but only inside a private work directory.

### Testing

The original `testing/test.sh` logic is converted into a generated staged script named:

```text
runs/cache/testing/<testing_name>__<testing_hash>/work/testing/run_test_cli.sh
```

This generated script is based on `test.sh`, but it is customized per testing artifact so `ss` mode skips reverse-complement/merge and `ds` mode runs them. The original repo-level `testing/test.sh` is not edited during a pipeline run.

## 7. Single-Step Commands

Train one artifact:

```bash
python -m idea.cli train --config configs/input_settings.yaml --training-set train_1hlo_1nkp_1nlw
```

Test one artifact:

```bash
python -m idea.cli test --config configs/input_settings.yaml --testing-set test_1hlo
```

Calculate energy directly from explicit files:

```bash
python -m idea.cli energy \
  --gamma runs/cache/training/<training_hash>/gamma_filtered.txt \
  --phi runs/cache/testing/<testing_hash>/phi_decoys.txt \
  --out runs/cache/energy/manual_test
```

The older energy script also supports explicit paths now:

```bash
python energy_calculation/calculate_testing_energy.py \
  --gamma path/to/gamma_filtered.txt \
  --phi path/to/phi_decoys.txt \
  --out path/to/output_dir
```
