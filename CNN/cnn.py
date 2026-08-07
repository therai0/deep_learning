import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim 


import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d( 3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d( 32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d( 64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
    
        self.model = nn.Sequential(
            nn.Linear(4*4*128,256),
            nn.ReLU(),

            nn.Linear(256,10)
        )

    def forward(self,x):
        x = self.layers(x)
        x = x.view(x.size(0),-1) # flatten layers 
        x = self.model(x)

if __name__ == "__main__":

    # creating transforms
    tarnsform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    # train and test dataset
    train = CIFAR10(root="./cnn_data",train=True,download=True,transform=tarnsform)
    test = CIFAR10(root="./cnn_data",train=False,download=True,transform=tarnsform)


    train_loader = DataLoader(train,batch_size=50,shuffle=True)
    test_loader = DataLoader(test,batch_size=50)

    cnn = CNN()
    optimizer = optim.Adam(cnn.parameters())
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    trian_loss = []

    for epoch in range(epochs):
        cnn.train()
        train_loss_per_epoch = 0
        for image,label in train_loader:
            optimizer.zero_grad()
            output = cnn.forward(image) # forward propagation
            loss = criterion(output,label)
            loss.backward() # backward propagation
            optimizer.step() # update parameters
            train_loss_per_epoch += loss 

        print(f"{epoch}/{epochs} loss ==> {train_loss_per_epoch}")


    # for evaluating the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images,label in test_loader:
            output = cnn.forward(images)
            _,predict = torch.max(output,1)
            correct = (predict == label).sum().item()
            total += label.size(0)

    print(f"{correct} out of {total}")