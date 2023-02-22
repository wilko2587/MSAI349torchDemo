# make a normalizer
import torch
import pandas as pd
from torch.utils.data import Dataset

class MinMaxScaler:
    def __init__(self, base_dataset):
        self.Xmin = torch.min(base_dataset.X, dim=0).values
        self.Xmax = torch.max(base_dataset.X, dim=0).values
        self.ymin = torch.min(base_dataset.y, dim=0).values
        self.ymax = torch.max(base_dataset.y, dim=0).values

    def normalize(self, dataset):
        dataset.X = (dataset.X - self.Xmin) / (self.Xmax - self.Xmin)
        dataset.y = (dataset.y - self.ymin) / (self.ymax - self.ymin)

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

normalizer = MinMaxScaler(train_data)
normalizer.normalize(train_data)
normalizer.normalize(valid_data)
normalizer.normalize(test_data)