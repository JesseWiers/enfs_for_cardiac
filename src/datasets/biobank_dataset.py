import numpy as np
from torch.utils.data import Dataset
import torch
import os 
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import logging

class BiobankEndpoints(Dataset):
        
    def __init__(
        self,
        root: str, 
        split: str='train',
        transform: torch.nn.Module=None,
        target_transform: torch.nn.Module=None,
        debug_mode: bool=False,
        debug_subset_size: int=100,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.data_seg = []
        print("Debug mode: ", debug_mode)
        patient_paths = sorted(os.listdir(root))
        total_patients = len(patient_paths)

        if debug_mode:
            subset = min(debug_subset_size, total_patients)
            patient_paths = patient_paths[:subset]

        if len(patient_paths) <= 1:
            split_index = len(patient_paths)
        else:
            split_index = int(0.8 * len(patient_paths))
            split_index = max(1, min(len(patient_paths) - 1, split_index))

        if split == 'train':
            patient_paths = patient_paths[:split_index]
            print(f"Biobank NIFTI: num_patients train = {len(patient_paths)}")
        elif split == 'test':
            patient_paths = patient_paths[split_index:]
            print(f"Biobank NIFTI: num_patients test = {len(patient_paths)}")
                
        
        for patient_path in patient_paths:
            nifti_file_path = os.path.join(root, patient_path, "cropped_sa.nii.gz")
            nifti_file_path_seg = os.path.join(root, patient_path, "cropped_seg_sa.nii.gz")

            if not os.path.exists(nifti_file_path) and not os.path.exists(nifti_file_path_seg):
                nifti_file_path = os.path.join(root, patient_path, "sa.nii.gz")
                nifti_file_path_seg = os.path.join(root, patient_path, "seg_sa.nii.gz")

                if not os.path.exists(nifti_file_path) and not os.path.exists(nifti_file_path_seg):
                    print(f"Skipping {patient_path}: NIFTI files not found.")
                    continue
                       
            # Load the NIFTI file
            nifti_image = nib.load(nifti_file_path)
            image_data = nifti_image.get_fdata()  # Shape: [H, W, Z, T]
                 
            nifti_image_seg = nib.load(nifti_file_path_seg)
            image_data_seg = nifti_image_seg.get_fdata()  # Shape: [H, W, Z, T]

            pad_config = ((3, 3), (0, 0), (0, 0), (0, 0))
            image_data = np.pad(
                image_data,
                pad_width=pad_config,
                mode='constant',
                constant_values=0,
            )
            image_data_seg = np.pad(
                image_data_seg,
                pad_width=pad_config,
                mode='constant',
                constant_values=0,
            )
            H, W, Z, T = image_data.shape

            if H != 77 or W != 77:
                print(f"Skipping {patient_path}: unexpected spatial shape ({H}, {W})")
                continue
            0
            # Iterate over timesteps
            for t in range(T):
                for z in range(Z):
                    image = image_data[:, :, z, t] # Load image 
                    image_seg = image_data_seg[:, :, z, t] # Load segmentation
                    
                    # if image_seg is all zeros, skip
                    if np.all(image_seg == 0):
                        continue
                    
                    # Min-max normalization per slice
                    slice_min = np.min(image)
                    slice_max = np.max(image)
                    
                    image = (image - slice_min) / (slice_max - slice_min)
                
                    # Assert all values are between 0 and 1
                    assert np.all(image >= 0) and np.all(image <= 1), f"Normalization failed: values outside [0,1] range for patient {patient_path}, t={t}, z={z}"
             
                    image = image[..., np.newaxis] # Add axis
                    
                    # Add original image
                    self.data.append(image)
                    self.data_seg.append(image_seg)

        print(f"Number of datapoints: {len(self.data)}")
        self.indices = np.arange(len(self.data))  # Add this line to store indices
                

    def __getitem__(self, index: int):
        """Returns a tuple of (data, target, index) for the given index"""
        return self.data[index], self.data_seg[index]

    def __len__(self):
        return len(self.data)
    
    