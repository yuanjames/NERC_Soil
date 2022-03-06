import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shap
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import minmax_scale, StandardScaler

X = pd.read_excel("Raw data for EDCs NSIS2 MA.xlsx", sheet_name=2).iloc[:, 1::]
y = pd.read_excel("Raw data for EDCs NSIS2 MA.xlsx", sheet_name=0).iloc[0:183, :][
    "DEHP"
]

index_drop = X[X["Elevation"] == -9999.0].index
X.drop(index_drop, inplace=True)
y.drop(index_drop, inplace=True)

feat_names = X.columns.values
ss = StandardScaler()
X = ss.fit_transform(X.values)

print(X)
print(y)

y.replace("<0.05", 0.05, inplace=True)

print(y)
y  = np.log(y.values)
print(min(y), max(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

X_train_summary = shap.kmeans(X_train, 10)

X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train.reshape(-1, 1))
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test.reshape(-1, 1))
dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(87, 100),
    nn.Sigmoid(),
    nn.Linear(100, 200),
    nn.Sigmoid(),
    nn.Linear(200, 64),
    nn.Sigmoid(),
    nn.Linear(64, 32),
    nn.Sigmoid(),
    nn.Linear(32, 1),

)
# model.cuda()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss(reduction="sum")
for e in range(10000):
    print(e)
    for step, (x, y) in enumerate(dataloader):
        model.zero_grad()
        y_pred = model(x)
        loss = torch.sqrt(loss_fn(y_pred, y))
        loss.backward()
        optimizer.step()
    print(torch.sqrt(loss_fn(model(X_test_tensor), y_test_tensor)).item())
# model.cpu()

with torch.no_grad():
    print(torch.sqrt(loss_fn(model(X_test_tensor), y_test_tensor)).item())


def f(x):
    return model(torch.Tensor(x)).detach().numpy().reshape(-1)


ex = shap.KernelExplainer(f, X_train_summary)
# shap_values = ex.shap_values(X_test.iloc[0, :])
# shap.force_plot(ex.expected_value, shap_values[0], X_test.iloc[0, :], matplotlib=True)
X_test = pd.DataFrame(data=X_test, columns=feat_names)

shap_values = ex.shap_values(X_test.iloc[0:10, :])
shap.summary_plot(shap_values, X_test.iloc[0:10, :])
