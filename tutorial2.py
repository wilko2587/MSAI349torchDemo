# add some validation loss, and train for multiple epochs

import torch
from torch import nn
import pandas as pd

data_pandas = pd.read_csv('./california_housing_train.csv')
data_torch = torch.Tensor(data_pandas.to_numpy())
X = data_torch[:, :-1]
y = data_torch[:, -1]

# min max norm
X = (X - X.min(dim=0).values)/(X.max(dim=0).values - X.min(dim=0).values)
y = (y - y.min())/(y.max() - y.min())

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

# split validation and train
X_train, y_train = X[0:10000], y[0:10000]
X_valid, y_valid = X[10000:], y[10000:]

# a better way
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True)

for epoch in range(50):
  out_train = net(X_train)
  loss = loss_func(out_train.flatten(), y_train)
  loss.backward() # load the gradients onto the network
  optimizer.step()

  with torch.no_grad():
    out_valid = net(X_valid)
    valid_loss = loss_func(out_valid.flatten(), y_valid)

  print("epoch {} train loss {:.3f} valid loss {:.3f}".format(epoch, loss.item(), valid_loss.item()))