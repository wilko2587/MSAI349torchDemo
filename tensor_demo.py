import torch
import pandas as pd

data_pandas = pd.DataFrame({"feature1": [1, 2, 3, 4, 5],
                             "feature2": [6, 7, 8, 9, 10],
                             "feature3": [11, 12, 13, 14, 15],
                             "feature4": [16, 17, 18, 19, 20],
                             })

print(data_pandas)

data_torch = torch.Tensor(data_pandas.to_numpy())

# squeeze, unsqueeze, view
#print(data_torch.shape)
#data_torch = data_torch.unsqueeze(1)
#print(data_torch.shape)
#data_torch = data_torch.squeeze(1)
#print(data_torch.shape)