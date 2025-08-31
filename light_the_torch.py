import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import timm

import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dir = "./data/train"
test_dir = "./data/test"
valid_dir = "./data/valid"

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    

target_to_class = {v:k for k,v in ImageFolder(train_dir).class_to_idx.items()}
target_to_class = {v:k for k,v in ImageFolder(test_dir).class_to_idx.items()}
target_to_class = {v:k for k,v in ImageFolder(valid_dir).class_to_idx.items()}

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

testset = PlayingCardDataset(test_dir,transform)
trainset = PlayingCardDataset(test_dir,transform)
validset = PlayingCardDataset(test_dir,transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle = True)
testloader = DataLoader(testset, batch_size=32, shuffle = False)
validloader = DataLoader(validset, batch_size=32, shuffle = False)

class SimpleCardClassifer(nn.Module):
    def __init__(self, num_classes = 53):
        super(SimpleCardClassifer, self).__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)
        

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    
num_epoch = 5
train_losses, valid_losses = [], []

model = SimpleCardClassifer(num_classes=53)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(num_epoch):
    model.train()
    running_loss =0.0
    for images,labels in tqdm(trainloader, desc="Training loop"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item() * labels.size(0)
    train_loss = running_loss / len(trainloader.dataset)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images,labels in tqdm(validloader, desc="Validation loop"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion (outputs, labels)
            running_loss += loss.item() * labels.size(0)
    validloss = running_loss /len(validloader.dataset)
    valid_losses.append(validloss)
    print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {train_loss}, Valid loss: {validloss}")

fails = []
with torch.no_grad():
    for i, (images,labels) in enumerate(tqdm(testloader, desc="Testing Model")):
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        loss = criterion (outputs, labels)
        running_loss += loss.item() * labels.size(0)
        for j, (output, label) in enumerate(zip(outputs,labels)):
            chosen = output.argmax().item()
            label = label.item()
            if chosen != label:
                fails.append(i*32+j)
    test_loss = running_loss / len(testloader.dataset)
    print(f"Test loss {test_loss}")
