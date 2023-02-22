# make a dataset class
import torch
import pandas as pd
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, datafile):
        data = pd.read_csv(datafile).astype('float')
        X_pandas = data.iloc[:, :-1]
        y_pandas = data.iloc[:, -1]

        self.X = torch.tensor(X_pandas.to_numpy())
        self.y = torch.tensor(y_pandas.to_numpy())

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


train_data = MyDataset(datafile='./california_housing_train.csv')
valid_data = MyDataset(datafile='./california_housing_valid.csv')
test_data = MyDataset(datafile='./california_housing_test.csv')