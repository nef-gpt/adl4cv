import torch
from torch.utils.data import Dataset
import h5py

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, dataset_name):
        self.hdf5_file = hdf5_file
        self.dataset_name = dataset_name
        with h5py.File(self.hdf5_file, 'r') as f:
            self.dataset_length = f[self.dataset_name].shape[0]
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            data = f[self.dataset_name][idx]
            return torch.tensor(data)