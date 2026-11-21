# LQR optimization (TPE)

```bash
.venv/bin/python scripts/optimize_lqr.py config.single.json
```

Tunes all four (single pole) or six (double pole) Q weights and R on a log scale. Results are saved without changing the input config.

| Flag | Default | Description |
|---|---|---|
| `--trials` | `100` | Number of additional trials. |
| `--seeds` | `42 43 44` | Training noise seeds. |
| `--validation-seeds` | `1042 1043 1044` | Separate validation noise seeds. |
| `--validation-configs` | Training configs | Config paths for validation. |
| `--search-factor` | `100` | Search each initial Q entry and R ÷factor to ×factor. |
| `--startup-trials` | `10` | Startup trials before adaptive TPE. |
| `--sampler-seed` | `42` | Optimizer random seed. |
| `--output` | `results/lqr-tpe-TIMESTAMP` | Result directory. |
| `--resume` | Off | Continue the study in `--output` with matching settings. |
| `-h`, `--help` | — | Show command help. |
