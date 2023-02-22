# read some data, make a network, and do one training step of a neural network

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

class FeedForward(nn.Module):  # module
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
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

out = net(X)
loss = loss_func(out.flatten(), y)
loss.backward()  # load the gradients onto the network
optimizer.step()

print(loss)

out = net(X)
loss = loss_func(out.flatten(), y)
loss.backward()  # load the gradients onto the network
optimizer.step()

print(loss)

out = net(X)
loss = loss_func(out.flatten(), y)
loss.backward()  # load the gradients onto the network
optimizer.step()

print(loss)