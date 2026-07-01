# Neural simulation-based inference

den_pie learns the posterior $p(\boldsymbol{\theta}\mid\boldsymbol{d})$ directly
from forward-model simulations using neural posterior estimation (the `sbi`
library with normalizing flows). Two pipelines infer the *same* parameters from
different representations of the data.

## Density pipeline ({mod}`den_pie.density`)

Conditions the flow on the **raw 3-D field**:

1. **Encoder** (`build_encoder`) — `cnn3d`, `swin3d`, or `resnet18` compresses
   the field to a summary vector.
2. **Flow** (`build_sbi_density_estimator`) — an `sbi` `maf`/`nsf` posterior with
   the encoder as its embedding network.
3. **Prior** (`build_composite_prior`) — uniform on cosmology, normal on bias.
4. **Training** (`DensityTraining`) — three stages: `encoder` (MSE pretrain),
   `flow` (freeze encoder, train the transform), `both` (end-to-end NLL).

## Spectra pipeline ({mod}`den_pie.spectra`)

Conditions the flow on the **summary vector** $P_{0,2,4}(k) + B_0$:

1. **Data** (`SpectraDataLoader`) — assembles the feature vector from the
   `pk_l_*`/`bk_0_*` arrays, applies per-channel transforms
   (`log` for $P_0$, `arcsinh` otherwise) and a per-feature train-split z-score.
   Datasets are resolved through `DATASET_REGISTRY`.
2. **Embedding** (`build_embedding`) — `none` (identity), `mlp` (trainable), or
   `pca` (frozen, whitened).
3. **Flow / prior / training** — an `sbi` `maf`/`nsf` posterior trained
   single-stage by `SpectraTraining`.

## Diagnostics

Both pipelines produce corner plots (test and fiducial), simulation-based
calibration (SBC) rank histograms, and TARP coverage. Posterior samples are
saved as `.npz` (`samples`, `label`) — the same files the
{doc}`comparison <comparison>` tool overlays on a Fisher forecast.

## Running

```bash
python -m den_pie spectra-train  params/spectra_train_fli_bias_SN_PCA.yaml
python -m den_pie spectra-plot   params/spectra_train_fli_bias_SN_PCA.yaml

python -m den_pie density-train  params/density_bias_train_stage2_flow_maf_HR_both.yaml
python -m den_pie density-plot   params/density_bias_train_stage2_flow_maf_HR_both.yaml
```
