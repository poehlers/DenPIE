"""Data loader for the spectra pipeline.

Reads per-simulation .npz files that already contain power-spectrum multipoles
(`pk_l_*`) and bispectrum monopole (`bk_0_*`), assembles a single normalized
feature vector per sim, and returns train/val/test/fiducial tensors.
"""
import logging
import glob
import os

import numpy as np
import torch

from .registry import DATASET_REGISTRY


# Map field name in user config -> suffix used in the .npz keys.
FIELD_SUFFIX = {
    "delta_ic": "ic",
    "delta_biased": "fin",
}

# Map multipole index (0, 2, 4) -> row in pk_l_<suffix>. Row 0 is the k-bin
# centers; rows 1, 2, 3 are P_0, P_2, P_4.
MULTIPOLE_ROW = {0: 1, 2: 2, 4: 3}


def _transform_multipole(x: torch.Tensor, l: int, eps: float = 1e-8) -> torch.Tensor:
    """Apply per-multipole transform. log for P_0, arcsinh otherwise."""
    if l == 0:
        return torch.log(x + eps)
    return torch.asinh(x)


def _transform_bk(x: torch.Tensor) -> torch.Tensor:
    return torch.asinh(x)


# Fiducial sims store no per-sim params; their truth label is constant. Mirrors
# the dict used in density/eval.py.
FIDUCIAL_COSMO = {
    'Omega_m': 0.3175, 'Omega_b': 0.0490, 'h': 0.6711, 'n_s': 0.9624,
    'sigma_8': 0.8340, 'b1': 1.0, 'b2': 0.0, 'bs2': 0.0, 'bn2': 0.0,
}


class SpectraDataLoader:
    """Load per-sim spectra, assemble feature vectors, split, and normalize."""

    def __init__(self, params: dict):
        self.cfg = params['spectra']['data']
        self.data = self.load()

    def load(self) -> dict:
        cfg = self.cfg
        dataset_key = cfg['dataset']
        if dataset_key not in DATASET_REGISTRY:
            raise KeyError(
                f"Unknown dataset '{dataset_key}'. Known keys: "
                f"{list(DATASET_REGISTRY.keys())}"
            )
        ds = DATASET_REGISTRY[dataset_key]

        fields = list(cfg['fields'])
        multipoles = list(cfg['multipoles'])
        use_bispectrum = bool(cfg.get('use_bispectrum', True))
        active_params = list(cfg['active_params'])
        val_split = cfg.get('val_split', 0.1)
        test_split = cfg.get('test_split', 0.1)
        n_realizations = cfg.get('n_realizations', None)
        k_max = cfg.get('k_max', None)

        for f in fields:
            if f not in FIELD_SUFFIX:
                raise KeyError(
                    f"Unknown field '{f}'. Known fields: {list(FIELD_SUFFIX.keys())}"
                )
        for l in multipoles:
            if l not in MULTIPOLE_ROW:
                raise KeyError(
                    f"Multipole {l} not in {{0, 2, 4}}"
                )

        # Resolve active-param column indices into the (param_keys concatenated)
        # vector.
        all_param_names = list(ds['param_names'])
        active_indices = [all_param_names.index(p) for p in active_params]

        # Determine which raw rows to read from each field's pk_l_* array.
        pk_rows = [MULTIPOLE_ROW[l] for l in multipoles]

        # Locate per-sim files.
        npz_files = sorted(glob.glob(os.path.join(ds['path'], "*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {ds['path']}")
        if n_realizations is not None and n_realizations < len(npz_files):
            npz_files = npz_files[:n_realizations]
        logging.info(
            f"spectra: dataset={dataset_key}, found {len(npz_files)} sims, "
            f"fields={fields}, multipoles={multipoles}, "
            f"use_bispectrum={use_bispectrum}"
        )

        # Probe the first file for k-bins, n_tri.
        d0 = np.load(npz_files[0])
        n_k_full = d0[f"pk_l_{FIELD_SUFFIX[fields[0]]}"].shape[1]
        k_bins = d0[f"pk_l_{FIELD_SUFFIX[fields[0]]}"][0]
        if k_max is not None:
            n_k = int((k_bins <= k_max).sum())
        else:
            n_k = n_k_full
        n_tri = d0[f"bk_0_{FIELD_SUFFIX[fields[0]]}"].shape[0]
        logging.info(
            f"  n_k={n_k}/{n_k_full} (k_max={k_max}), n_tri={n_tri}"
        )

        # Pass 1: load all spectra into a single tensor, screen NaNs, collect
        # params.  Spectra are small (~few hundred floats per sim), so we hold
        # the whole training corpus in RAM.
        raw_blocks, params_all, valid_indices = self._scan_and_load(
            npz_files, fields, pk_rows, multipoles, use_bispectrum,
            n_k, n_tri, ds['param_keys'], len(all_param_names),
        )
        N = len(valid_indices)
        if N < len(npz_files):
            logging.warning(
                f"Skipped {len(npz_files) - N} NaN files out of {len(npz_files)}"
            )
        for j, pname in enumerate(all_param_names):
            logging.info(
                f"  {pname}: [{params_all[:, j].min():.4f}, "
                f"{params_all[:, j].max():.4f}]"
            )

        # Apply per-channel transforms (still un-normalized).
        raw_blocks = self._apply_transforms(
            raw_blocks, fields, multipoles, use_bispectrum, n_k, n_tri,
        )
        features_all = torch.cat(raw_blocks, dim=1)  # (N, F)
        F = features_all.shape[1]
        logging.info(f"  feature_dim={F}")

        # Train/val/test split — prefix train (so stats are reproducible),
        # mirroring density/data.py.
        n_test = int(test_split * N)
        n_val = int(val_split * N)
        n_train = N - n_val - n_test
        y_all = params_all[:, active_indices]

        x_train = features_all[:n_train]
        x_val = features_all[n_train:n_train + n_val]
        x_test = features_all[n_train + n_val:]
        y_train = y_all[:n_train]
        y_val = y_all[n_train:n_train + n_val]
        y_test = y_all[n_train + n_val:]

        # Pass 2: fit per-feature z-score on training split, apply to all.
        feat_mean = x_train.mean(dim=0, keepdim=True)
        feat_std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        x_train = (x_train - feat_mean) / feat_std
        x_val = (x_val - feat_mean) / feat_std
        x_test = (x_test - feat_mean) / feat_std

        norm_fid_data = None
        y_fiducial = None
        fiducial_path = ds.get('fiducial_path')
        if fiducial_path:
            norm_fid_data, y_fiducial = self._load_fiducial(
                fiducial_path, fields, pk_rows, multipoles, use_bispectrum,
                n_k, n_tri, feat_mean, feat_std,
                active_param_names=active_params,
                n_max=max(n_test, 100),
            )

        logging.info(
            f"  split: train={n_train}, val={n_val}, test={n_test}, "
            f"fiducial={0 if norm_fid_data is None else norm_fid_data.shape[0]}"
        )

        return {
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test,
            'norm_fid_data': norm_fid_data,
            'y_fiducial': y_fiducial,
            'feature_dim': F,
            'feature_mean': feat_mean,
            'feature_std': feat_std,
            'k_bins': torch.from_numpy(k_bins[:n_k]).float(),
        }

    @staticmethod
    def _scan_and_load(npz_files, fields, pk_rows, multipoles, use_bispectrum,
                       n_k, n_tri, param_keys, n_all_params,
                       read_params: bool = True):
        """Pass 1: load raw spectra blocks per field.

        Returns a list of un-transformed blocks (one tensor per field), all of
        shape (N, block_dim), the parameter tensor (or None if
        ``read_params`` is False), and the indices of valid (non-NaN) files.
        """
        per_field_pk = [torch.empty((len(npz_files), len(pk_rows), n_k),
                                    dtype=torch.float32) for _ in fields]
        per_field_bk = [torch.empty((len(npz_files), n_tri), dtype=torch.float32)
                        for _ in fields] if use_bispectrum else None

        params_list = []
        valid_indices = []
        for i, fpath in enumerate(npz_files):
            d = np.load(fpath)
            has_nan = False
            for f in fields:
                suf = FIELD_SUFFIX[f]
                pk = d[f"pk_l_{suf}"][pk_rows, :n_k]
                if np.isnan(pk).any():
                    has_nan = True
                    break
                if use_bispectrum:
                    bk = d[f"bk_0_{suf}"][:, 3]
                    if np.isnan(bk).any():
                        has_nan = True
                        break
            if has_nan:
                logging.warning(
                    f"Skipping {os.path.basename(fpath)}: contains NaN"
                )
                continue
            for k, f in enumerate(fields):
                suf = FIELD_SUFFIX[f]
                per_field_pk[k][i] = torch.from_numpy(
                    d[f"pk_l_{suf}"][pk_rows, :n_k]
                ).float()
                if use_bispectrum:
                    per_field_bk[k][i] = torch.from_numpy(
                        d[f"bk_0_{suf}"][:, 3]
                    ).float()
            if read_params:
                parts = [torch.from_numpy(d[k]).float() for k in param_keys]
                p = torch.cat(parts) if len(parts) > 1 else parts[0]
                params_list.append(p)
            valid_indices.append(i)
            if (i + 1) % 1000 == 0:
                logging.info(f"  scan: {i + 1}/{len(npz_files)}")

        if not valid_indices:
            raise RuntimeError("No valid (non-NaN) sim files")

        # Trim invalid rows out of pre-allocated tensors.
        idx_t = torch.tensor(valid_indices, dtype=torch.long)
        per_field_pk = [t.index_select(0, idx_t) for t in per_field_pk]
        if use_bispectrum:
            per_field_bk = [t.index_select(0, idx_t) for t in per_field_bk]

        if read_params:
            params_all = torch.stack(params_list, dim=0)
            assert params_all.shape[1] == n_all_params, (
                f"Param tensor has {params_all.shape[1]} cols, "
                f"expected {n_all_params} from param_names"
            )
        else:
            params_all = None

        # Pack into a list of (N, ?) blocks: per-field pk then per-field bk.
        raw_blocks = []
        for k in range(len(fields)):
            raw_blocks.append(per_field_pk[k])  # (N, n_l, n_k)
        if use_bispectrum:
            for k in range(len(fields)):
                raw_blocks.append(per_field_bk[k])  # (N, n_tri)
        return raw_blocks, params_all, valid_indices

    @staticmethod
    def _apply_transforms(raw_blocks, fields, multipoles, use_bispectrum,
                          n_k, n_tri):
        """Per-channel log/arcsinh, then flatten each block to (N, dim)."""
        out = []
        n_fields = len(fields)
        for k in range(n_fields):
            pk_block = raw_blocks[k]  # (N, n_l, n_k)
            # Per-multipole transform.
            transformed = torch.empty_like(pk_block)
            for li, l in enumerate(multipoles):
                transformed[:, li] = _transform_multipole(pk_block[:, li], l)
            out.append(transformed.reshape(transformed.shape[0], -1))
        if use_bispectrum:
            for k in range(n_fields):
                bk_block = raw_blocks[n_fields + k]  # (N, n_tri)
                out.append(_transform_bk(bk_block))
        return out

    @staticmethod
    def _load_fiducial(fiducial_path, fields, pk_rows, multipoles, use_bispectrum,
                       n_k, n_tri, feat_mean, feat_std,
                       active_param_names, n_max):
        """Load fiducial sims and normalize with training stats.

        Fiducial sims share constant parameters (Planck-like cosmology + identity
        bias) and don't store them per-file; the truth label is taken from
        FIDUCIAL_COSMO.
        """
        npz_files = sorted(glob.glob(os.path.join(fiducial_path, "*.npz")))
        if not npz_files:
            logging.warning(
                f"No fiducial .npz files found in {fiducial_path}, "
                f"skipping fiducial set"
            )
            return None, None
        npz_files = npz_files[:n_max]
        raw_blocks, _, _ = SpectraDataLoader._scan_and_load(
            npz_files, fields, pk_rows, multipoles, use_bispectrum,
            n_k, n_tri, param_keys=None, n_all_params=0, read_params=False,
        )
        transformed = SpectraDataLoader._apply_transforms(
            raw_blocks, fields, multipoles, use_bispectrum, n_k, n_tri,
        )
        x = torch.cat(transformed, dim=1)
        x = (x - feat_mean) / feat_std
        for p in active_param_names:
            if p not in FIDUCIAL_COSMO:
                raise KeyError(
                    f"FIDUCIAL_COSMO missing entry for active param '{p}'"
                )
        truth = torch.tensor(
            [FIDUCIAL_COSMO[p] for p in active_param_names], dtype=torch.float32,
        )
        y = truth.unsqueeze(0).expand(x.shape[0], -1).contiguous()
        logging.info(
            f"  loaded {x.shape[0]} fiducial sims from {fiducial_path}"
        )
        return x, y
