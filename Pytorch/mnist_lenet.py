import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# three kernels + two pooling layers
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5, 1)
#         self.conv2 = nn.Conv2d(6, 16, 5, 1)
#         self.conv3 = nn.Conv2d(16, 120, 4, 1)
#         self.fc1 = nn.Linear(120, 64)
#         self.fc2 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.sigmoid(x)
#         x = F.max_pool2d(x, 2)
#         x = self.conv2(x)
#         x = F.sigmoid(x)
#         x = F.max_pool2d(x, 2)
#         x = self.conv3(x)
#         # reshape the 4D tensor to 2D tensor
#         x = x.view(x.shape[0], -1)
#         x = self.fc1(x)
#         x = F.sigmoid(x)
#         x = self.fc2(x)
#         return x
#
# net = LeNet()

# two kernels + three fully connected layers
net = nn.Sequential(
    nn.Conv2d(1, 6, 5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(6, 16, 5),
    nn.Sigmoid(),
    nn.AvgPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

# in the pytorch, the input shape is (batch_size, channels, height, width)
X = torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

transform = transforms.Compose([
    # transform the image to tensor and normalize it
    transforms.ToTensor(),
    # 0.1307 is the mean and 0.3081 is the standard deviation of the MNIST dataset
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

loss_function = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loss_function.to(device)

# create the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# train the model
def train(epoch_id):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        X, y = data
        if use_cuda:
            X, y = X.cuda(), y.cuda()

        # x -> y_hat
        optimizer.zero_grad()
        # y_hat = loss
        loss = loss_function(net(X), y)
        # loss -> grads
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch_id + 1, batch_idx + 1, running_loss))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on the test set: %.2f %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(20):
        train(epoch_id=epoch)
        if epoch % 2 == 0:
            test()
