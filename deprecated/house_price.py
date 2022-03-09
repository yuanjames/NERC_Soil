from matplotlib.backend_bases import MouseEvent
from matplotlib.pyplot import axis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import shap
import time
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

start_time = time.time()
df = pd.read_csv('kc_house_data.csv')
print(df)

X = df.drop(['id', 'date', 'price'], axis=1)
y = df.iloc[:, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

X_train_summary = shap.kmeans(X_train.to_numpy(), 10)

X_train_tensor = torch.Tensor(X_train.to_numpy())
y_train_tensor = torch.Tensor(y_train.to_numpy().reshape(-1, 1))
dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

model = nn.Sequential(
          nn.Linear(18, 200),
          nn.ReLU(),
          nn.Linear(200, 1),
          nn.ReLU()
        )
# model.cuda()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss(reduction='sum')
for e in range(200):
    print(e)
    for step, (x, y) in enumerate(dataloader):
        model.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
print(time.time()-start_time)
# model.cpu()
time.sleep(2)
def f(x):
    return model(torch.Tensor(x)).detach().numpy().reshape(-1)
    

ex = shap.KernelExplainer(f, X_train_summary)
# shap_values = ex.shap_values(X_test.iloc[0, :])
# shap.force_plot(ex.expected_value, shap_values[0], X_test.iloc[0, :], matplotlib=True)

shap_values = ex.shap_values(X_test.iloc[0:40, :])
shap.summary_plot(shap_values, X_test.iloc[0:40, :])