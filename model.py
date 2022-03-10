from torch import nn, optim
from torch.nn import functional as F
import torch
from sklearn.metrics import r2_score
class Net(nn.Module):
    def __init__(self, num_feats=8):
        super(Net, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(num_feats, 50),
            nn.Sigmoid(),
            nn.Dropout(0.25),
            nn.Linear(50, 100),
            nn.Sigmoid(),
            nn.Dropout(0.25),
            nn.Linear(100, 15),
            nn.Sigmoid(),
            nn.Linear(15, 1),
        )

    def forward(self, x):
        y = self.fc_layers(x)
        return y

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.sqrt(F.mse_loss(output, target)).item() # sum up batch loss
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))