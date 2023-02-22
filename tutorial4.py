# adding in dataset and dataloader

import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataset(Dataset):
  def __init__(self, datafile):
    data = pd.read_csv(datafile)
    X_pandas = data.iloc[:, :-1]
    y_pandas = data.iloc[:, -1]
    self.X = torch.tensor(X_pandas.to_numpy()).float()
    self.y = torch.tensor(y_pandas.to_numpy()).float()

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return len(self.X)


class MinMaxScaler:
  def __init__(self, base_dataset):
    self.Xmin = torch.min(base_dataset.X, dim=0).values
    self.Xmax = torch.max(base_dataset.X, dim=0).values
    self.ymin = torch.min(base_dataset.y, dim=0).values
    self.ymax = torch.max(base_dataset.y, dim=0).values

  def normalize(self, dataset):
    dataset.X = (dataset.X - self.Xmin) / (self.Xmax - self.Xmin)
    dataset.y = (dataset.y - self.ymin) / (self.ymax - self.ymin)

  def denormalize(self, y):
    return y * (self.ymax - self.ymin) + self.ymin


class FeedForward(nn.Module): # module
  def __init__(self):
    super(FeedForward, self).__init__()
    self.linear1 = nn.Linear(8, 32)
    self.relu1 = nn.LeakyReLU()
    self.linear2 = nn.Linear(32, 16)
    self.relu2 = nn.LeakyReLU()
    self.linear_out = nn.Linear(16, 1)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu1(x)
    x = self.linear2(x)
    x = self.relu2(x)
    x = self.linear_out(x)
    return x


net = FeedForward()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

train_data = MyDataset(datafile='./california_housing_train.csv')
valid_data = MyDataset(datafile='./california_housing_valid.csv')
test_data = MyDataset(datafile='./california_housing_test.csv')

normalizer = MinMaxScaler(train_data)
normalizer.normalize(train_data)
normalizer.normalize(valid_data)
normalizer.normalize(test_data)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)

for epoch in range(50):
  train_losses = []
  valid_losses = []

  for X_train, y_train in train_loader:
    out_train = net(X_train)
    loss = loss_func(out_train.flatten(), y_train)
    loss.backward() # load the gradients onto the network
    optimizer.step()
    train_losses.append(loss.item())

  with torch.no_grad():
    for X_valid, y_valid in valid_loader:
      out_valid = net(X_valid)
      valid_loss = loss_func(out_valid.flatten(), y_valid)
      valid_losses.append(valid_loss.item())

  train_loss_epoch = np.sum(train_losses)/len(train_losses)
  valid_loss_epoch = np.sum(valid_losses)/len(valid_losses)
  print("epoch {} train loss {:.3f} valid loss {:.3f}".format(epoch, train_loss_epoch, valid_loss_epoch))

sample = test_data[0][0]
median_price = net(sample)
print('median px: ', median_price)