"""Dataset registry for the spectra pipeline.

Each entry maps a dataset key to the on-disk location and parameter metadata.
Per-sim .npz files in `path` (and `fiducial_path`) are expected to contain:
  - pk_l_<field>: (4, n_k)   row 0 = k bin centers, rows 1/2/3 = P_0/P_2/P_4
  - bk_0_<field>: (n_tri, 4) columns 0..2 = (k1, k2, k3), column 3 = B_0
  - <param_key>: 1-D arrays whose concatenation matches `param_names`
"""


DATASET_REGISTRY = {
    "fli_bias_SN": {
        "path": "/projects/prjs1926/data/fli_data/fli_bias_SN/samples_dir/prior/",
        "fiducial_path": "/projects/prjs1926/data/fli_data/fli_fiducial_bias_SN/samples_dir/prior/",
        "param_keys": ["cosmo_params", "bias_params"],
        "param_names": ["Omega_m", "sigma_8", "b1", "b2", "bs2", "bn2"],
        "param_priors": [
            ("uniform", 0.1, 0.5),   # Omega_m
            ("uniform", 0.6, 1.0),   # sigma_8
            ("normal",  1.0, 0.5),   # b1
            ("normal",  0.0, 2.0),   # b2
            ("normal",  0.0, 2.0),   # bs2
            ("normal",  0.0, 2.0),   # bn2
        ],
        "box_size": 1000.0,
        "los_axis": 2,
    },
}
