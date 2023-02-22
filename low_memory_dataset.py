# make a dataset class that read data from disk on-the-fly
# not fully complete!

import torch
import pandas as pd
from torch.utils.data import Dataset
import os

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    def __getitem__(self, index):
        sample_data = pd.read_csv(self.file_names[index], index_col=0)
        X = sample_data.iloc[:-1].astype('float')
        y = sample_data.iloc[-1].astype('float')
        X_torch = torch.Tensor(X.to_numpy())
        y_torch = torch.Tensor(y.to_numpy())
        return X_torch[index], y_torch[index]

    def __len__(self):
        return len(self.file_names)


train_data = MyDataset(data_dir='./data/train')