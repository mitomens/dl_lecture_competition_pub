import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob

import torch
from scipy.signal import resample, butter, filtfilt


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", preprocess=None) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path))
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))

            if self.preprocess:
                for p in self.preprocess:
                    X = p(X)
            
            return X, y, subject_idx
        else:
            if self.preprocess:
                for p in self.preprocess:
                    X = p(X)
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]

class Resample:
    def __init__(self, new_rate):
        self.new_rate = new_rate

    def __call__(self, X):
        X_resampled = resample(X, self.new_rate, axis=-1)
        return torch.from_numpy(X_resampled).float()

class Filter:
    def __init__(self, lowcut, highcut, fs, order=5):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def butter_bandpass(self):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def __call__(self, X):
        b, a = self.butter_bandpass()
        X_filtered = filtfilt(b, a, X.numpy(), axis=-1)
        X_filtered = np.ascontiguousarray(X_filtered)
        return torch.from_numpy(X_filtered).float()

class BaselineCorrection:
    def __call__(self, X):
        baseline = X.mean(dim=-1, keepdim=True)
        X_corrected = X - baseline
        return X_corrected

class Scaling:
    def __call__(self, X):
        return (X - X.mean()) / X.std()