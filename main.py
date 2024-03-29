from preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, StandardScaler
from model import Net, train, test
import torch
from torch.utils.data import DataLoader, TensorDataset
import shap
from torch.nn import functional as F

# Part 1 data preprocessing
X, y, feat_names= extraction_data(feature=[1, 19], task='PBDE 47')

# X = minmax_scale(X.values, axis=0)
# y  = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)


X_train_tensor = torch.Tensor(X_train.values)
y_train_tensor = torch.Tensor(y_train.values.reshape(-1, 1))
X_test_tensor = torch.Tensor(X_test.values)
y_test_tensor = torch.Tensor(y_test.values.reshape(-1, 1))

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=8, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=8, shuffle=True)

# Part 2 create NNs
model = Net(num_feats =feat_names.shape[0])
device = torch.device('cpu')
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
for epoch in range(1,  350):
    train(model, device, train_loader, optimizer, epoch)
    # test(model, device, test_loader)

with torch.no_grad():
    predict = model(X_test_tensor)
    loss = torch.sqrt(F.mse_loss(predict, y_test_tensor)).item()
    print('RMSE:', loss)
    
    

# Part 3 explain the NNs
def f(x):
    return model(torch.Tensor(x)).detach().numpy().reshape(-1)

# plot 1
e = shap.DeepExplainer(model, X_train_tensor)
shap_values = e.shap_values(X_test_tensor)
print(type(shap_values))
processed_shap_values = np.sum(shap_values[:,8::], axis=1)
shap_values = shap_values[:, 0:9]
shap_values[:, -1] = processed_shap_values
feat_names[8] = 'LandCoverXa'
shap.summary_plot(shap_values, X_test_tensor.numpy()[:, 0:9], feature_names = feat_names[0:9])

# # plot 2
# explainer = shap.Explainer(f, X_train.values)
# shap_values = explainer(X_test)
# shap.plots.bar(shap_values)
# shap.plots.heatmap(shap_values)
