import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob

import torch


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
            
            return X, y, subject_idx
        else:
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]

"""
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
"""
"""
class Scaling:
    def __call__(self, n, axis=None, ddof=0):
        # 平均値を計算
        mean_n = torch.mean(n, axis = axis, keepdims = True)
        # 標準偏差を計算。ddofが0ならば標準偏差、ddofが1ならば不標準偏差
        # keepdims:配列の次元数を落とさずに結果を求めるための引数
        std_n = torch.std(n, axis = axis, keepdims = True, ddof = ddof)
        # 標準化の計算
        standard_n = (n - mean_n) / std_n
    
        return standard_n
"""

class Scheduler:
    def __init__(self, epochs, lr, warmup_length=5):
        """
        Arguments
        ---------
        epochs : int
            学習のエポック数．
        lr : float
            学習率．
        warmup_length : int
            warmupを適用するエポック数．
        """
        self.epochs = epochs
        self.lr = lr
        self.warmup = warmup_length

    def __call__(self, epoch):
        """
        Arguments
        ---------
        epoch : int
            現在のエポック数．
        """
        progress = (epoch - self.warmup) / (self.epochs - self.warmup)
        progress = np.clip(progress, 0.0, 1.0)
        lr = self.lr * 0.5 * (1. + np.cos(np.pi * progress))

        if self.warmup:
            lr = lr * min(1., (epoch+1) / self.warmup)

        return lr
    
def set_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

