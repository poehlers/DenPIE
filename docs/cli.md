# Command-line interface

Everything is driven through `python -m den_pie <subcommand>`. Training/plotting
verbs take a YAML param card; the forward-model and Fisher verbs are thin
pass-throughs to their underlying tools.

| Subcommand | Purpose |
|------------|---------|
| `density-train` / `density-plot` / `density-search` | Train / evaluate / Optuna-search the field-level SBI model |
| `spectra-train` / `spectra-plot` | Train / evaluate the $P_\ell + B_0$ SBI model |
| `forward-sample` | Generate forward-model SBI training data (wraps `falcon sample prior`) |
| `fisher-forecast` | Run a Fisher forecast (wraps `den_pie.fisher.run_fisher`) |
| `fisher-compare` | Overlay an SBI posterior on a Fisher forecast corner |

## Inference

```bash
python -m den_pie spectra-train  params/spectra_train_fli_bias_SN_PCA.yaml
python -m den_pie spectra-plot   params/spectra_train_fli_bias_SN_PCA.yaml
python -m den_pie density-train   params/density_bias_train_stage2_flow_maf_HR_both.yaml
python -m den_pie density-plot    params/density_bias_train_stage2_flow_maf_HR_both.yaml
python -m den_pie density-search  params/optuna_search.yaml
```

## Forward model

```bash
python -m den_pie forward-sample --config-name config_files/config_base.yml --run-dir RUN_DIR
```

All arguments after `forward-sample` are forwarded verbatim to
`falcon sample prior`.

## Fisher

```bash
python -m den_pie fisher-forecast --config config_files/fisher/fisher_5param_P024_B0_emp.yml
python -m den_pie fisher-forecast -h        # full flag set (delegated to the runner)
```

## Comparison

```bash
python -m den_pie fisher-compare <fisher_run> <sbi_run> --out sbi_vs_fisher.png
```

:::{note}
`forward-sample` and `fisher-forecast` are *delegating* verbs — den_pie
intercepts them before its own argument parser and forwards the remaining
arguments to the underlying tool, whose full flag set is available via
`<verb> -h`.
:::
