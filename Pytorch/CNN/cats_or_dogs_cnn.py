import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import torch.utils.data

# check if CUDA is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# create the model
class Net(nn.Module):
    def __int__(self):
        super(Net, self).__int__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        # the first parameter is the number of input channels
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # return the logit
        return x


# create the model
model = Net().to(device)

# Use data augmentation
transforms_train = transforms.Compose([
    # resize the image to 150 * 150
    transforms.Resize((150, 150)),


    transforms.RandomRotation(40), # rotate the image 40 degree
    transforms.RandomHorizontalFlip(), # flip the image horizontally
    # transforms.RandomVerticalFlip(), # flip the image vertically
    # transforms.RandomCrop(150, padding=10), # crop the image
    # transforms.RandomResizedCrop(150), # resize the image
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # change the brightness, contrast, saturation of the image
    # transforms.RandomAffine(degrees=40, translate=(0.1, 0.1), scale=(0.2, 0.2), shear=0.2), # affine transformation

    # make the image data from 0 to 1(normalize), and transform it to tensor
    transforms.ToTensor()
])

transforms_test = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# get the current script path
current_script_path = os.path.abspath(__file__)

# get the project root
project_root = os.path.dirname(os.path.dirname(current_script_path))

# the dataset path
# wait for the dataset to download
base_dir = os.path.join(project_root, 'datasets', 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_dataset = datasets.ImageFolder(train_dir, transform=transforms_train)
validation_dataset = datasets.ImageFolder(validation_dir, transform=transforms_test)

print(train_dataset.classes)
print(train_dataset.class_to_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=20, num_workers=4)

# start training
# the data(Net) is the logit, so we need to use the CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()

# there are three types of linked layer:
# torch.nn.Linear + torch.sigmoid = torch.nn.BCELoss # binary classification, 1 linear layer
# torch.nn.Linear + BCEWithLogitLoss # multi classification, 1 linear layer
# torch.nn.Linear + torch.log_softmax = torch.nn.NLLLoss # multi classification, 2 linear layer

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        X, y = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # get the index of the max log-probability
        pred = y_pred.argmax(dim=1, keepdim=True)
        # get the number of correct predictions
        running_corrects += pred.eq(y.view_as(pred)).sum().item()

    #print the information
    epoch_loss = running_loss * 20 / len(train_dataset)
    epoch_acc = 100 * running_corrects / len(train_dataset)
    print('Train Epoch: %d, Loss: %.4f, Accuracy: %.2f' % (epoch, epoch_loss, epoch_acc))

    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            X, y = data.to(device), target.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(validation_dataset)
    test_acc = 100 * correct / len(validation_dataset)
    print('Test set: Average loss: %.4f, Accuracy: %.2f' % (test_loss, test_acc))