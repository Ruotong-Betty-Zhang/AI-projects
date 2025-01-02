import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
print(use_cuda)

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# define the transformation
# transforms.ToTensor() convert the image to tensor
# transforms.Normalize() normalize the image
# 0.1307 and 0.3081 are the mean and standard deviation of the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# load the datasets
diabetes1 = datasets.MNIST('../Pytorch/data', train=True, download=True, transform=transform)
diabetes2 = datasets.MNIST('../Pytorch/data', train=False, download=True, transform=transform)

# create the data loaders
train_loader = torch.utils.data.DataLoader(diabetes1, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(diabetes2, batch_size=1000)


# for batch_idx, data in enumerate(train_loader, 0):
#     inputs, labels = data
#     # view(-1, 28*28) means reshape the tensor to 28*28, from (60000, 1, 28, 28) to (60000, 28*28)
#     x = inputs.view(-1, 28*28)
#     x_std = x.std().item()
#     x_mean = x.mean().item()
#
# print('mean: ' + str(x_mean))
# print('std: ' + str(x_std))

# create the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # fully connected layer
        self.fc1 = nn.Linear(2304, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # pooling kernel size 2 * 2
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # 4D tensor to 2D tensor
        x = torch.flatten(x, 1)
        # fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# create the model instance
model = Net().to(device)


# define the training logic
def train_step(data, target, model, optimizer):
    # set the gradients to zero
    optimizer.zero_grad()
    # forward pass
    output = model(data)
    # calculate the loss
    loss = F.nll_loss(output, target)
    # backward pass
    loss.backward()
    # update the weights
    optimizer.step()
    return loss


# define the testing logic
def test_step(data, target, model, test_loss, correct):
    # get the output
    output = model(data)
    # calculate the loss
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    # get the index of the max log-probability
    pred = output.argmax(dim=1, keepdim=True)
    # get the number of correct predictions
    correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss, correct


# create the optimizer for training
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = train_step(data, target, model, optimizer)
        # print the information every 10 batches
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            test_loss, correct = test_step(data, target, model, test_loss, correct)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset))
    )

