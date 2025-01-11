import torch
import torch.nn as nn
import os
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torchvision.models import alexnet, AlexNet_Weights
import time

# path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# img_path = os.path.join(path, "datasets", "cats_and_dogs_filtered", "validation", "cats","cat.2000.jpg")
# img = read_image(img_path)
#
# # get the weights of the AlexNet model
# weights = AlexNet_Weights.DEFAULT
# # create the model
# model = alexnet(weights=weights)
# # print the structure of the model
# tmp = model.eval()
# print(tmp)
#
# # create the preprocessing function
# preprocess = weights.transforms()
# batch = preprocess(img).unsqueeze(0)
#
# prediction = model(batch).squeeze(0).softmax(0)
# print(prediction)
# print(len(prediction))
# # now, prediction is a tensor of size 1000 containing the probabilities of each class
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# print(f"Class ID: {class_id}")
# print(f"Score: {score}")
#
# # meta data includes the information about the classes
# category_name = weights.meta['categories'][class_id]
# print(f"Category: {category_name}")
#
# class_to_index = {cls: idx for (idx, cls) in enumerate(weights.meta['categories'])}
# print(class_to_index)

# check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights = AlexNet_Weights.DEFAULT
model = alexnet(weights=weights)

for k, v in model.named_parameters():
    if not k.startswith('classifier'):
        # if the layer is not the classifier layer, set requires_grad to False
        v.requires_grad = False
    # print(k, v.requires_grad)

# replace the classifier layer with a new one
model.classifier[1] = nn.Linear(9216, 4096)
model.classifier[4] = nn.Linear(4096, 4096)
model.classifier[6] = nn.Linear(4096, 2)

model = model.to(device)
transforms_for_train = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

    transforms.ToTensor(),
])

transforms_for_validation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
validation_path = os.path.join(path, "datasets", "cats_and_dogs_filtered", "validation")
train_path = os.path.join(path, "datasets", "cats_and_dogs_filtered", "train")

train_dataset = datasets.ImageFolder(train_path, transform=transforms_for_train)
validation_dataset = datasets.ImageFolder(validation_path, transform=transforms_for_validation)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=20, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)
        running_corrects += pred.eq(target.view_as(pred)).sum().item()

    epoch_loss = running_loss * 20 / len(train_dataset)
    epoch_acc = running_corrects * 100 / len(train_dataset)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}, Time: {time.time() - start_time}")

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss = test_loss * 20 / len(validation_dataset)
    test_accuracy = correct * 100 / len(validation_dataset)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
