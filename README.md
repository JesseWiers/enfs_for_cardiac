# Geometry-Aware Cardiac MRI Representation Learning with Equivariant Neural Fields

**Jesse L. Wiers, David R. Wessels, Lukas P.A. Arts, Samuel Ruiperez-Campillo, Maarten Z.H. Kolk, Fleur V.Y. Tjong\*, Erik J. Bekkers\***

*Accepted at MIDL 2026 — Medical Imaging with Deep Learning.*

This repository contains the code for the paper **"Geometry-Aware Cardiac MRI Representation Learning with Equivariant Neural Fields"**, which evaluates Equivariant Neural Fields (ENFs) for cardiac MRI reconstruction and downstream clinical prediction.

---

## Overview

The pipeline consists of three steps:

1. **Train a 2D reconstructer** (`train_2d_reconstructer.py`) — learn ENF or CNF model weights on short-axis CMR slices using MAML.
2. **Extract latent point clouds** (`create_latent_dataset.py`) — run the trained ENF over the full dataset and store per-patient latent point clouds in an HDF5 file.
3. **Endpoint classification** (`endpoint_classification.py`) — train a Transformer or EGNN classifier on the stored latents for clinical endpoint prediction.

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd enfs_for_cardiac

# 2. Create a virtual environment (conda or venv)
conda create -n enf_cardiac python=3.10
conda activate enf_cardiac

# 3. Install JAX with CUDA support first (see https://jax.readthedocs.io/en/latest/installation.html)
#    Adjust cuda12 → cuda11 if your cluster uses CUDA 11.x
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Data

Experiments use short-axis cardiac cine MRI from the [UK Biobank](https://www.ukbiobank.ac.uk/). Access requires an approved application.

Expected data structure after preprocessing with the [ukbb_cardiac](https://github.com/baiwenjia/ukbb_cardiac) segmentation tool:

```
<dataset_root>/
  <patient_id>/
    cropped_sa.nii.gz        # cropped short-axis cine volume [H, W, Z, T]
    cropped_seg_sa.nii.gz    # corresponding segmentation mask
```

Update the `root` path in `src/datasets/__init__.py` and `config.dataset.root` in the scripts to point to your data directory.

---

## Usage

All scripts use [ml_collections](https://github.com/google/ml_collections) for configuration. Flags can be overridden on the command line with `--config.key=value`.

### 1. Train the ENF reconstructer

```bash
python scripts/train_2d_reconstructer.py \
  --config.run_name=enf_64x32 \
  --config.exp_name=my_experiment \
  --config.recon_enf.num_latents=64 \
  --config.recon_enf.latent_dim=32 \
  --config.train.batch_size=16 \
  --config.train.num_epochs_train=100 \
  --config.save_checkpoints=True
```

Checkpoints are saved under `model_checkpoints/<exp_name>/<run_name>/`.

Key config options:

| Flag | Default | Description |
|------|---------|-------------|
| `config.bi_invariant` | `translational` | Symmetry group: `translational` or `rotational` |
| `config.recon_enf.num_latents` | `16` | Number of latent points K |
| `config.recon_enf.latent_dim` | `64` | Latent context dimension d_c |
| `config.optim.inner_steps` | `3` | MAML inner loop steps |
| `config.optim.inner_lr` | `(2., 30., 0.)` | Inner LR for (pose, context, window) |

### 2. Extract latent point clouds

```bash
python scripts/create_latent_dataset.py \
  --config.dataset.root=/path/to/cropped_cmr \
  --config.dataset.latent_dataset_path=/path/to/output/latents.h5 \
  --config.recon_enf.checkpoint_path=/path/to/checkpoint.pkl \
  --config.recon_enf.num_latents=64 \
  --config.recon_enf.latent_dim=32
```

Latents for all patients are stored in an HDF5 file. Already-processed patients are skipped on re-runs.

### 3. Train an endpoint classifier

```bash
python scripts/endpoint_classification.py \
  --config.dataset.latent_path=/path/to/latents.h5 \
  --config.dataset.endpoint=heart_failure \
  --config.model.name=transformer \
  --config.train.num_epochs=100 \
  --config.exp_name=endpoint_classification
```

Supported endpoints (must match keys in the HDF5 `/endpoints/` group):
`cardiomyopathy`, `sudden_cardiac_death`, `all_cause_mortality`, `heart_failure`, `myocardial_infarction`, `atrial_fibrillation`, `ischaemic_heart_disease`.

Key config options:

| Flag | Default | Description |
|------|---------|-------------|
| `config.model.name` | `transformer` | Classifier: `transformer` or `egnn` |
| `config.model.hidden_size` | `768` | Model hidden dimension |
| `config.model.transformer_depth` | `12` | Number of transformer blocks |
| `config.dataset.z_indices` | `3` | Number of middle SAX slices to use |
| `config.dataset.t_indices` | `(0,10,20,30,40,49)` | Cardiac phases to include |

---

## Repository Structure

```
enfs_for_cardiac/
├── scripts/
│   ├── train_2d_reconstructer.py       # Step 1: ENF/CNF reconstruction training
│   ├── create_latent_dataset.py        # Step 2: Latent extraction
│   └── endpoint_classification.py     # Step 3: Downstream endpoint prediction
├── src/
│   ├── enf/
│   │   ├── model.py                    # EquivariantNeuralField (Flax)
│   │   ├── bi_invariants.py            # TranslationBI, RotoTranslationBI2D
│   │   └── utils.py                    # Latent initialisation, coordinate grids
│   ├── datasets/
│   │   ├── biobank_dataset.py          # UK Biobank NIfTI dataset (for training)
│   │   └── biobank_latent_endpoint_dataset.py  # HDF5 latent dataset (for classification)
│   └── downstream/
│       ├── transformer_enf.py          # Transformer classifier for latent point clouds
│       └── egnn.py                     # EGNN classifier for latent point clouds
└── requirements.txt
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wiers2026enf,
  title     = {Geometry-Aware Cardiac {MRI} Representation Learning with Equivariant Neural Fields},
  author    = {Wiers, Jesse L. and Wessels, David R. and Arts, Lukas P.A. and
               Ruiperez-Campillo, Samuel and Kolk, Maarten Z.H. and
               Tjong, Fleur V.Y. and Bekkers, Erik J.},
  booktitle = {Medical Imaging with Deep Learning (MIDL)},
  year      = {2026},
}
```

---

## Acknowledgements

This work builds on [Equivariant Neural Fields](https://arxiv.org/abs/2406.05753) by Wessels et al. (2024) and uses CMR data from the [UK Biobank](https://www.ukbiobank.ac.uk/).
