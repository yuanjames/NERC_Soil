import imp
from torch import nn, optim
from torch.nn import functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(86, 100),
            nn.Sigmoid(),
            nn.Linear(100, 200),
            nn.Sigmoid(),
            nn.Linear(200, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
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

# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.mse_loss(output.log(), target).item() # sum up batch loss
#             pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)))