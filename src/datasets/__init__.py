import torch
import torchvision
from PIL import Image
import numpy as np

from src.datasets.biobank_dataset import BiobankEndpoints

def image_to_numpy(image: Image) -> np.ndarray:
    """
    Convert a PIL image to a numpy array.
    """
    return np.array(image) / 255


def numpy_collate(batch: list[np.ndarray]) -> np.ndarray:
    """
    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle_train: bool = True,
    debug_mode: bool = False,
    debug_subset_size: int = 50,
):
    """ 
    Returns dataloaders for the requested dataset.

    Args:
        dataset_name: Name of the dataset configuration to load.
        batch_size: Batch size for both train and test loaders.
        num_workers: Number of workers per DataLoader.
        seed: Seed applied to the PyTorch generator and worker init.
        shuffle_train: Whether to shuffle the training loader.
        debug_mode: Whether to limit dataset size for faster iterations.
        debug_subset_size: Maximum number of patients when debug_mode is True.
    """    
    
    # Create generator with seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    if dataset_name == "biobank_endpoints":
        transforms = torchvision.transforms.Compose([image_to_numpy])
        train_dset = BiobankEndpoints(
            root='/projects/prjs1252/biobank_endpoint_dataset/nifti_dataset',
            split="train",
            transform=transforms,
            debug_mode=debug_mode,
            debug_subset_size=debug_subset_size,
        )
        test_dset = BiobankEndpoints(
            root='/projects/prjs1252/biobank_endpoint_dataset/nifti_dataset',
            split="test",
            transform=transforms,
            debug_mode=debug_mode,
            debug_subset_size=debug_subset_size,
        )
    else:
        raise NotImplementedError("Dataset not implemented yet.")
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=numpy_collate,
        drop_last=True,
        num_workers=num_workers,
        generator=generator,  
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id) 
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
        drop_last=True,
        num_workers=num_workers,
        generator=generator, 
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)  
    )

    return train_loader, test_loader

