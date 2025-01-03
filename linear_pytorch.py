from PIL import Image
import matplotlib.pyplot as plt

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim 

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="./data"
        positive="Positive"
        negative="Negative"

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:10000] 
            self.Y=self.Y[0:10000] 
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)    
  
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image=Image.open(self.all_files[idx])
        y=self.Y[idx]
       
        if self.transform:
            image = self.transform(image)
        image = image.view(-1)
        return image, y
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform =transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean, std)])
dataset_train=Dataset(transform=transform,train=True)
dataset_val=Dataset(transform=transform,train=False)

dataset_train[0][0].shape
size_of_image=3*227*227
print(size_of_image)

torch.manual_seed(0)

learning_rate = 0.1
momentum_term =0.1
batch = 5
criterion = nn.CrossEntropyLoss()

epochs = 5




import torch 
import torch.nn as nn

import torch.nn.functional as F

class LinearCNN(nn.Module):
    def __init__(self):
        super(LinearCNN,self).__init__()
        self.input_shape = 3*227*227
        self.conv1 = nn.Conv1d(in_channels = 1,out_channels=8,kernel_size=3, stride=1)
        self.flatten = nn.Flatten() #將卷積出來的結果運用在dense
        self.fc = nn.Linear(8*(3 * 227 * 227 - 2),2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = x.view(x.size(0), 1, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

model = LinearCNN()
optimizer =  torch.optim.SGD(model.parameters(),lr = learning_rate,momentum=momentum_term)
train_loader = DataLoader(dataset_train,batch_size=batch)
val_loader = DataLoader(dataset_val,batch_size=batch)

train_losses = []
val_losses = []
train_acc_ = []
val_acc_ = []

for epoch in range(epochs):
    print(f"in model train {epoch}")
    model.train()
    running_loss = 0.0
    correct = 0
    for x,y in train_loader:
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        correct += (torch.argmax(yhat, dim=1) == y).float().sum()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct / len(dataset_train)
    train_acc_.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    val_acc = 0
    with torch.no_grad():
        for x,y in val_loader:
            yhat = model(x)
            loss = criterion(yhat,y)
            val_loss += loss.item()
            val_acc += (torch.argmax(yhat, dim=1) == y).float().sum()
        
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_acc / len(dataset_val)
    val_losses.append(val_loss)
    val_acc_.append(val_acc)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Epoch {epoch + 1}, Train ACC: {train_accuracy:.4f}, Val ACC: {val_acc:.4f}')

    
plt.figure(figsize=(10,5))

