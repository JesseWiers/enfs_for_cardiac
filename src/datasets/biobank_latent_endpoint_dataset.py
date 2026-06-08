import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class BiobankLatentEndpointDataset(Dataset):
    """Dataset of pre-extracted ENF latent point clouds and clinical endpoint labels.

    Expected HDF5 layout (produced by create_latent_dataset.py):

        /metadata                       — group with attrs: num_latents, latent_dim
        /patients/<patient_id>/
            p  (1, Z*T*K, pos_dim+2)   — stacked poses (t, z, x, y, ...)
            c  (1, Z*T*K, latent_dim)  — stacked context vectors
            g  (1, Z*T*K, 1)           — stacked window scalars
        /endpoints/<endpoint_name>/
            <patient_id>               — scalar label (0 or 1)

    Args:
        hdf5_path: Path to the HDF5 latent dataset.
        endpoint_name: Name of the clinical endpoint group in the HDF5 file.
        z_indices: Slice indices to keep. -1 keeps all; int keeps that many middle
                   slices; tuple keeps the listed indices.
        t_indices: Timepoint indices to keep. Same convention as z_indices.
        debug_limit: If set, cap the dataset to this many samples.
    """

    def __init__(self, hdf5_path, endpoint_name, z_indices=-1, t_indices=-1, debug_limit=None):
        self.hdf5_path = hdf5_path
        self.endpoint_name = endpoint_name
        self.z_indices = z_indices
        self.t_indices = t_indices

        self.patient_ids = []
        self.labels = []

        with h5py.File(hdf5_path, 'r') as f:
            if 'patients' not in f:
                raise KeyError("HDF5 file has no 'patients' group.")
            if 'endpoints' not in f or endpoint_name not in f['endpoints']:
                raise KeyError(f"Endpoint '{endpoint_name}' not found in HDF5 file.")

            endpoint_group = f['endpoints'][endpoint_name]
            patient_keys = list(f['patients'].keys())

            for pid in patient_keys:
                if pid in endpoint_group:
                    self.patient_ids.append(pid)
                    self.labels.append(int(endpoint_group[pid][()]))

        if debug_limit is not None:
            self.patient_ids = self.patient_ids[:debug_limit]
            self.labels = self.labels[:debug_limit]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = self.labels[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            group = f['patients'][pid]
            p = group['p'][0]  # (Z*T*K, pos_dim+2)
            c = group['c'][0]  # (Z*T*K, latent_dim)
            g = group['g'][0]  # (Z*T*K, 1)
            Z = group.attrs.get('Z', None)
            T = group.attrs.get('T', None)

        # Optionally sub-select slices and timepoints.
        # The stored p has leading dimensions [t, z, x, y] in the pose.
        # We filter by the t and z coordinate columns (indices 0 and 1).
        if Z is not None and T is not None:
            p, c, g = _filter_zt(p, c, g, Z, T, self.z_indices, self.t_indices)

        return pid, (p.astype(np.float32), c.astype(np.float32), g.astype(np.float32)), label


def _filter_zt(p, c, g, Z, T, z_indices, t_indices):
    """Keep only the rows of (p, c, g) that match the requested z and t indices."""
    t_lin = np.linspace(-1, 1, T)
    z_lin = np.linspace(-1, 1, Z)

    def _resolve(indices, linspace):
        n = len(linspace)
        if indices == -1:
            return linspace
        if isinstance(indices, int):
            mid = n // 2
            half = indices // 2
            sel = list(range(max(0, mid - half), min(n, mid + half + (indices % 2))))
            return linspace[sel]
        return linspace[list(indices)]

    t_vals = set(np.round(_resolve(t_indices, t_lin), 6).tolist())
    z_vals = set(np.round(_resolve(z_indices, z_lin), 6).tolist())

    t_col = np.round(p[:, 0], 6)
    z_col = np.round(p[:, 1], 6)

    mask = np.array([t in t_vals and z in z_vals for t, z in zip(t_col, z_col)])
    return p[mask], c[mask], g[mask]


def _numpy_collate(batch):
    patient_ids, z_tuples, labels = zip(*batch)
    p = np.stack([z[0] for z in z_tuples])
    c = np.stack([z[1] for z in z_tuples])
    g = np.stack([z[2] for z in z_tuples])
    return list(patient_ids), (p, c, g), np.array(labels)


def create_dataloaders(
    hdf5_path,
    endpoint_name,
    batch_size,
    train_split=0.75,
    val_split=0.15,
    num_workers=0,
    debug_limit=None,
    random_seed=42,
    z_indices=-1,
    t_indices=-1,
    train_subset_fraction=1.0,
):
    """Create train / val / test DataLoaders from the latent HDF5 dataset.

    Args:
        hdf5_path: Path to the latent dataset HDF5 file.
        endpoint_name: Clinical endpoint to predict.
        batch_size: Batch size for all loaders.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation (remainder → test).
        num_workers: DataLoader worker count.
        debug_limit: Cap total dataset size for quick runs.
        random_seed: Seed for reproducible splits.
        z_indices: Slice selection (see BiobankLatentEndpointDataset).
        t_indices: Timepoint selection (see BiobankLatentEndpointDataset).
        train_subset_fraction: Fraction of training set to actually use.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = BiobankLatentEndpointDataset(
        hdf5_path=hdf5_path,
        endpoint_name=endpoint_name,
        z_indices=z_indices,
        t_indices=t_indices,
        debug_limit=debug_limit,
    )

    n = len(dataset)
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n).tolist()

    n_train = int(train_split * n)
    n_val = int(val_split * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if train_subset_fraction < 1.0:
        keep = max(1, int(len(train_idx) * train_subset_fraction))
        train_idx = train_idx[:keep]

    def _make_loader(idx, shuffle):
        return DataLoader(
            Subset(dataset, idx),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_numpy_collate,
            drop_last=True,
        )

    return _make_loader(train_idx, shuffle=True), _make_loader(val_idx, shuffle=False), _make_loader(test_idx, shuffle=False)
