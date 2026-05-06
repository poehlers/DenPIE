import logging
import glob
import os
import gc

import numpy as np
import torch
from torch.utils.data import Dataset


class FLIDatasetLazy(Dataset):
    """Lazy 3D-field dataset: loads one npz per __getitem__ and applies
    log1p + per-voxel z-score normalization on the fly.

    Normalization statistics (``mean``, ``std``) must be precomputed from the
    training split and passed in. Channel count is inferred from
    ``field_keys``.
    """

    def __init__(self, npz_paths, indices, field_keys, mean, std,
                 params_tensor, log_transform=True):
        self.npz_paths = npz_paths
        self.indices = list(indices)
        self.field_keys = list(field_keys)
        # Store mean/std without the leading batch dim to avoid a squeeze per getitem
        self.mean = mean.squeeze(0)   # (C, res, res, res)
        self.std = std.squeeze(0)
        self.params = params_tensor   # (N, n_params) — already subset to active
        self.log_transform = log_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path = self.npz_paths[self.indices[i]]
        d = np.load(path)
        x = torch.stack(
            [torch.from_numpy(d[fk]).float() for fk in self.field_keys], dim=0
        )
        if self.log_transform:
            x.clamp_(min=-0.999).log1p_()
        x.sub_(self.mean).div_(self.std + 1e-8)
        return x, self.params[i]


class FLIDataLoader:
    """Loads FLI .npz files (3D density fields + cosmological parameters).

    Produces a dict with lazy train/val datasets and materialized test +
    fiducial tensors, so eval.py can index ``x_test`` as before.

    Resolution and channel count are inferred from the first npz.
    """

    def __init__(self, params: dict):
        self.cfg = params['density']['data']
        self.data = self.load()

    def load(self) -> dict:
        cfg = self.cfg
        folder = cfg['data_path']
        if 'field_keys' in cfg:
            field_keys = list(cfg['field_keys'])
        else:
            field_keys = [cfg['field_key']]
        C = len(field_keys)
        param_keys_list = cfg['param_keys']
        all_param_names = cfg['all_param_names']
        active_params = cfg['active_params']
        val_split = cfg.get('val_split', 0.1)
        test_split = cfg.get('test_split', 0.1)
        n_realizations = cfg.get('n_realizations', None)
        log_transform = cfg.get('log_transform', True)
        fiducial_path = cfg.get('fiducial_path', None)

        active_indices = [all_param_names.index(p) for p in active_params]
        n_all_params = len(all_param_names)

        npz_files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {folder}")
        logging.info(f"Found {len(npz_files)} FLI simulation files in {folder}")
        logging.info(f"Loading {C} field channel(s): {field_keys}")

        if n_realizations is not None and n_realizations < len(npz_files):
            npz_files = npz_files[:n_realizations]
            logging.info(f"Using first {n_realizations} realizations")

        # Pass 1: scan for NaNs, collect small per-file params
        valid_indices, params_all, res = self._scan_and_collect_params(
            npz_files, field_keys, param_keys_list, n_all_params
        )
        N = len(valid_indices)
        if N < len(npz_files):
            logging.warning(
                f"Skipped {len(npz_files) - N} NaN files out of {len(npz_files)}"
            )
        logging.info(f"Valid FLI simulations: {N}  (res={res}, C={C})")
        for j, pname in enumerate(all_param_names):
            logging.info(
                f"  {pname}: [{params_all[:, j].min():.4f}, "
                f"{params_all[:, j].max():.4f}]"
            )

        # Split (train/val/test) over the valid indices. Train is the prefix
        # so stats are computed from a fixed subset, reproducibly.
        n_test = int(test_split * N)
        n_val = int(val_split * N)
        n_train = N - n_val - n_test
        train_idx = valid_indices[:n_train]
        val_idx = valid_indices[n_train:n_train + n_val]
        test_idx = valid_indices[n_train + n_val:]

        # Active-param tensors per split
        y_all = params_all[:, active_indices]
        y_train = y_all[:n_train]
        y_val = y_all[n_train:n_train + n_val]
        y_test = y_all[n_train + n_val:]

        # Pass 2: streaming mean/std over the train split only.
        mean, std = self._streaming_stats(
            npz_files, train_idx, field_keys, res, log_transform
        )
        for c, fk in enumerate(field_keys):
            logging.info(
                f"  normalization [{fk}]: "
                f"mean={mean[0, c].mean().item():.4f}, "
                f"std={std[0, c].mean().item():.4f}"
            )

        # Pass 3: materialize test (small — 10% of dataset).
        x_test = self._materialize(
            npz_files, test_idx, field_keys, mean, std, log_transform
        )

        # Lazy train/val datasets
        train_ds = FLIDatasetLazy(
            npz_files, train_idx, field_keys, mean, std,
            y_train, log_transform=log_transform,
        )
        val_ds = FLIDatasetLazy(
            npz_files, val_idx, field_keys, mean, std,
            y_val, log_transform=log_transform,
        )

        norm_fid_data = None
        if fiducial_path:
            norm_fid_data = self._load_fiducial(
                n_test, mean, std, fiducial_path, field_keys, log_transform
            )

        del params_all, y_all
        gc.collect()

        logging.info(f"Data split: train={n_train}, val={n_val}, test={n_test}")
        return {
            'x_train': train_ds,
            'y_train': y_train,
            'x_val': val_ds,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test,
            'norm_fid_data': norm_fid_data,
            'field_mean': mean,
            'field_std': std,
        }

    @staticmethod
    def _scan_and_collect_params(npz_files, field_keys, param_keys_list,
                                 n_all_params):
        """Pass 1: check NaNs, collect params, return resolution.

        Reads each npz once. Keeps only the small param arrays in RAM;
        releases the large field arrays immediately.
        """
        test_shape = np.load(npz_files[0])
        res = test_shape[field_keys[0]].shape[0]

        valid_indices = []
        params_list = []
        for i, fpath in enumerate(npz_files):
            d = np.load(fpath)
            has_nan = False
            for fk in field_keys:
                arr = d[fk]
                if np.isnan(arr).any():
                    has_nan = True
                    break
            if has_nan:
                logging.warning(
                    f"Skipping {os.path.basename(fpath)}: contains NaN"
                )
                continue
            parts = [torch.from_numpy(d[k]).float() for k in param_keys_list]
            p = torch.cat(parts) if len(parts) > 1 else parts[0]
            params_list.append(p)
            valid_indices.append(i)
            if (i + 1) % 1000 == 0:
                logging.info(f"  scan: {i + 1}/{len(npz_files)}")

        if not valid_indices:
            raise RuntimeError("No valid (non-NaN) FLI files")

        params_all = torch.stack(params_list, dim=0)
        assert params_all.shape[1] == n_all_params, (
            f"Param tensor has {params_all.shape[1]} cols, "
            f"expected {n_all_params} from all_param_names"
        )
        return valid_indices, params_all, res

    @staticmethod
    def _streaming_stats(npz_files, indices, field_keys, res, log_transform):
        """Pass 2: per-voxel (mean, std) over ``indices`` using running sums.

        Keeps two float64 accumulators of shape (1, C, res, res, res) —
        ~16 MB each at 128³ × 2 channels.
        """
        C = len(field_keys)
        s = torch.zeros((1, C, res, res, res), dtype=torch.float64)
        s2 = torch.zeros((1, C, res, res, res), dtype=torch.float64)
        n = 0
        logging.info(
            f"Computing normalization stats over {len(indices)} train files..."
        )
        for i in indices:
            d = np.load(npz_files[i])
            x = torch.stack(
                [torch.from_numpy(d[fk]).float() for fk in field_keys], dim=0
            ).unsqueeze(0)
            if log_transform:
                x.clamp_(min=-0.999).log1p_()
            xd = x.double()
            s.add_(xd)
            s2.add_(xd * xd)
            n += 1
            if n % 500 == 0:
                logging.info(f"  stats pass: {n}/{len(indices)}")
        mean_d = s / n
        var_d = (s2 / n - mean_d * mean_d).clamp_(min=0)
        mean = mean_d.float()
        std = var_d.sqrt().float()
        return mean, std

    @staticmethod
    def _materialize(npz_files, indices, field_keys, mean, std, log_transform):
        """Load all files at ``indices`` into a single normalized tensor."""
        C = len(field_keys)
        if not indices:
            return torch.empty((0, C, 0, 0, 0), dtype=torch.float32)
        res = np.load(npz_files[indices[0]])[field_keys[0]].shape[0]
        out = torch.empty((len(indices), C, res, res, res), dtype=torch.float32)
        mean_s = mean.squeeze(0)
        std_s = std.squeeze(0)
        for k, i in enumerate(indices):
            d = np.load(npz_files[i])
            x = torch.stack(
                [torch.from_numpy(d[fk]).float() for fk in field_keys], dim=0
            )
            if log_transform:
                x.clamp_(min=-0.999).log1p_()
            x.sub_(mean_s).div_(std_s + 1e-8)
            out[k] = x
        return out

    @staticmethod
    def _load_fiducial(n_fid_sims, train_mean, train_std,
                       fiducial_path, field_keys, log_transform=True):
        """Load FLI fiducial sims, normalize with training stats."""
        if isinstance(field_keys, str):
            field_keys = [field_keys]
        C = len(field_keys)

        npz_files = sorted(glob.glob(os.path.join(fiducial_path, "*.npz")))[:n_fid_sims]
        if not npz_files:
            raise FileNotFoundError(
                f"No FLI fiducial .npz files found in {fiducial_path}"
            )
        if len(npz_files) < n_fid_sims:
            logging.warning(
                f"Only {len(npz_files)} FLI fiducial sims available, "
                f"requested {n_fid_sims}"
            )

        # Skip NaN files (check inline to avoid preallocating then trimming)
        valid = []
        for fpath in npz_files:
            d = np.load(fpath)
            if any(np.isnan(d[fk]).any() for fk in field_keys):
                logging.warning(
                    f"Skipping fiducial {os.path.basename(fpath)}: contains NaN"
                )
                continue
            valid.append(fpath)

        fields = FLIDataLoader._materialize(
            valid, list(range(len(valid))), field_keys,
            train_mean, train_std, log_transform
        )
        logging.info(
            f"Loaded {len(valid)} valid FLI fiducial sims from "
            f"{fiducial_path} ({C} channel(s))"
        )
        return fields
