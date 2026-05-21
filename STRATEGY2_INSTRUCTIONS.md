# IDEA Strategy2 Ridge Refinement

Strategy2 fits a refined IDEA gamma from experimental binding data and then uses that gamma in the normal IDEA dot-product energy calculation.

## Inputs

Put Strategy2-specific inputs outside `testing/sequences/`, for example:

```text
data/strategy2/8K8D/dna_half.seq
data/strategy2/8K8D/exp.txt
```

Rules:

- `dna_half.seq`: one DNA motif per line.
- `exp.txt`: one numeric experimental value per line.
- The two files must have the same number of non-empty lines.
- The sequence length must match the configured template testing structure and `dna_mode`.

## Config

Use `configs/strategy2_settings.yaml` as the starting point. The three ridge choices are configurable:

```yaml
strategy2:
  defaults:
    alpha: 1.0          # number or auto
    target_sign: negate # negate -> y = -exp; identity -> y = exp
    fit_intercept: true # true or false
```

Each run references an existing testing set as the template used to generate phi:

```yaml
testing:
  - id: 8K8D
    dna_mode: ds

strategy2:
  runs:
    ridge_8K8D:
      template_testing: test_8K8D
      sequences: data/strategy2/8K8D/dna_half.seq
      exp: data/strategy2/8K8D/exp.txt
```

## Run

```bash
python -m idea.cli strategy2 --config configs/strategy2_settings.yaml --dry-run
python -m idea.cli strategy2 --config configs/strategy2_settings.yaml
```

Run one Strategy2 block:

```bash
python -m idea.cli strategy2 --config configs/strategy2_settings.yaml --run ridge_8K8D
```

Force recomputation:

```bash
python -m idea.cli strategy2 --config configs/strategy2_settings.yaml --force
```

## Outputs

Strategy2 writes to a separate root:

```text
runs_strategy2/cache/testing/<fit_run>__<hash>/phi_decoys.txt
runs_strategy2/cache/ridge/<run>__<hash>/gamma_refined.txt
runs_strategy2/cache/energy/<run>__<hash>/Energy_mg.txt
runs_strategy2/experiments/<run_id>/jobs_summary.csv
```

The final Strategy2 prediction output is `Energy_mg.txt`. No model-prediction or leave-one-out outputs are produced.
