"""
dataset.py

2.5D PyTorch Dataset for brain tumor resectability classification.

For each patient:
  - Finds tumor center of mass from segmentation mask
  - Extracts 5 slices per direction (center-2 to center+2)
  - Does this for 3 directions (axial, coronal, sagittal)
  - Does this for 4 modalities (T1, T1c, T2, FLAIR)
  - Stacks into tensor of shape [60, 224, 224]
  - Z-score normalises each channel
  - Random flips and rotations during training
"""

import os
import random
import warnings
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")

VOLUME_SIZE  = 128
SLICE_SIZE   = 224
N_SLICES     = 5
SLICE_OFFSET = N_SLICES // 2   # = 2
MODALITIES   = ["T1", "T1c", "T2", "FLAIR"]
N_CHANNELS   = len(MODALITIES) * 3 * N_SLICES   # 60


def find_file(folder, patient_id, suffix):
    exact = os.path.join(folder, f"{patient_id}_{suffix}.nii.gz")
    if os.path.isfile(exact):
        return exact
    for fname in os.listdir(folder):
        if fname.endswith(f"_{suffix}.nii.gz"):
            return os.path.join(folder, fname)
    return None


def load_volume(path, is_mask=False):
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if is_mask:
        data = (data > 0.5).astype(np.float32)
    return data


def resize_volume(volume, target=128):
    factors = tuple(target / s for s in volume.shape)
    return zoom(volume, factors, order=1).astype(np.float32)


def get_tumor_center(mask):
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    return tuple(int(round(c)) for c in coords.mean(axis=0))


def _resize_slice(slc, size):
    if slc.shape != (size, size):
        fy = size / slc.shape[0]
        fx = size / slc.shape[1]
        slc = zoom(slc, (fy, fx), order=1)
    return slc.astype(np.float32)


def extract_5slices(volume, center, size=224):
    """Extract 5 slices per direction through tumor center."""
    cx, cy, cz = center
    H, W, D    = volume.shape
    cx = int(np.clip(cx, 0, H - 1))
    cy = int(np.clip(cy, 0, W - 1))
    cz = int(np.clip(cz, 0, D - 1))

    all_slices = []
    for dz in range(-SLICE_OFFSET, SLICE_OFFSET + 1):
        iz = int(np.clip(cz + dz, 0, D - 1))
        all_slices.append(_resize_slice(volume[:, :, iz], size))
    for dy in range(-SLICE_OFFSET, SLICE_OFFSET + 1):
        iy = int(np.clip(cy + dy, 0, W - 1))
        all_slices.append(_resize_slice(volume[:, iy, :], size))
    for dx in range(-SLICE_OFFSET, SLICE_OFFSET + 1):
        ix = int(np.clip(cx + dx, 0, H - 1))
        all_slices.append(_resize_slice(volume[ix, :, :], size))

    return np.stack(all_slices, axis=0)   # [15, 224, 224]


def zscore(channel):
    mask = channel != 0
    if mask.sum() < 10:
        return np.zeros_like(channel)
    mean = channel[mask].mean()
    std  = channel[mask].std()
    if std < 1e-8:
        return np.zeros_like(channel)
    return np.clip((channel - mean) / std, -5.0, 5.0).astype(np.float32)


class TumorResectabilityDataset(Dataset):
    def __init__(self, labels_csv, data_dir, augment=False):
        self.data_dir = data_dir
        self.augment  = augment
        df = pd.read_csv(labels_csv)
        self.samples = list(zip(df["patient_id"].tolist(), df["label"].tolist()))

    def __len__(self):
        return len(self.samples)

    def get_sample_weights(self):
        labels  = np.array([label for _, label in self.samples])
        counts  = np.bincount(labels)
        weights = 1.0 / counts[labels]
        return weights.tolist()

    def _load_patient(self, patient_id, folder):
        seg_path = find_file(folder, patient_id, "tumor_segmentation")
        if seg_path:
            seg = load_volume(seg_path, is_mask=True)
            seg = resize_volume(seg, target=VOLUME_SIZE)
        else:
            seg = np.zeros((VOLUME_SIZE,) * 3, dtype=np.float32)

        center = get_tumor_center(seg)

        all_channels = []
        for mod in MODALITIES:
            path = find_file(folder, patient_id, mod)
            if path is None:
                all_channels.append(
                    np.zeros((15, SLICE_SIZE, SLICE_SIZE), dtype=np.float32)
                )
                continue
            vol    = load_volume(path)
            vol    = resize_volume(vol, target=VOLUME_SIZE)
            slices = extract_5slices(vol, center, size=SLICE_SIZE)
            for i in range(slices.shape[0]):
                slices[i] = zscore(slices[i])
            all_channels.append(slices)

        return np.concatenate(all_channels, axis=0)   # [60, 224, 224]

    def _augment(self, tensor):
        if random.random() > 0.5:
            tensor = TF.hflip(tensor)
        if random.random() > 0.5:
            tensor = TF.vflip(tensor)
        if random.random() > 0.5:
            angle  = random.uniform(-15, 15)
            tensor = TF.rotate(tensor, angle)
        return tensor

    def __getitem__(self, idx):
        patient_id, label = self.samples[idx]
        folder = os.path.join(self.data_dir, patient_id)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Patient folder not found: {folder}")
        volume = self._load_patient(patient_id, folder)
        tensor = torch.from_numpy(volume)
        if self.augment:
            tensor = self._augment(tensor)
        return tensor, torch.tensor(label, dtype=torch.float32)
