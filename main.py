from preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from model import Net, train
import torch
from torch.utils.data import DataLoader, TensorDataset
import shap

# Part 1
X, y, feat_names= extraction_data()

X = minmax_scale(X.values)
y  = np.log(y.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train.reshape(-1, 1))
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test.reshape(-1, 1))

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=5, shuffle=True)

# Part 2
model = Net()
num_epochs = 2
device = torch.device('cpu')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
for epoch in range(1,  101):
    train(model, device, train_loader, optimizer, epoch)
    
    
# Part 3
def f(x):
    return model(torch.Tensor(x)).detach().numpy().reshape(-1)

batch = next(iter(test_loader))
images, _ = batch

background = images[:2]
test_images = images[-2:-1]

e = shap.DeepExplainer(model, X_train_tensor)
shap_values = e.shap_values(X_test_tensor)
shap.summary_plot(shap_values, X_test)