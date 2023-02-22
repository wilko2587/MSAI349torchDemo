# plot a loss curve, and test the accuracy

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
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

train_data = MyDataset(datafile='./california_housing_train.csv')
valid_data = MyDataset(datafile='./california_housing_valid.csv')
test_data = MyDataset(datafile='./california_housing_test.csv')

train_loader = DataLoader(train_data, batch_size=32)
valid_loader = DataLoader(train_data, batch_size=32)

for epoch in range(100):
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
  print("epoch {} train loss {:.2f} valid loss {:.2f}".format(epoch, train_loss_epoch, valid_loss_epoch))

# now we have a trained network, run some stats