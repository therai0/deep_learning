from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import matplotlib.pyplot as plt 

class PreANN:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def pre_ann(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        X_test_scaled = scalar.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        return [train_loader,test_loader]


"""
Define model
"""
class ANN(nn.Module):
    def __init__(self,input_size):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            # first hidden layer
            nn.Linear(input_size, 6),
            nn.ReLU(),
            # second hidden layer
            nn.Linear(6, 6),
            nn.ReLU(),
            # this is the output layer
            nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
   
    df = pd.read_csv("data/powerplant_data.csv")
    X = df.drop(columns=["PE"])
    y = df["PE"]
    pre_ann = PreANN(X, y)
    train_loader,test_loader = pre_ann.pre_ann()


    ann_model = ANN(X.shape[1])
    mse = nn.MSELoss()
    optimizer = optim.Adam(ann_model.parameters())

    epochs = 100 # 100 times we are showing this data to models

    train_loss = []
    test_loss = []
    best_epoch_loss = float("inf")
    for epoch in range(epochs):
        ann_model.train()
        running_loss = 0.0
        for xb,yb in train_loader:
            optimizer.zero_grad()
    
            output = ann_model(xb) # forward propagation
            loss = mse(output,yb) # calculate the loss value
            loss.backward() # backward propagation compute the gradient descent
            running_loss += loss.item() 
            optimizer.step() # update the weights

        avg_loss_per_epoch = running_loss/ len(train_loader)
        train_loss.append(avg_loss_per_epoch)

        ann_model.eval() # now model switch to evaluation mode
        runing_val_loss = 0.0
        with torch.no_grad(): # no gradient compute
            for xb,yb in test_loader:
                output = ann_model(xb)
                loss = mse(output,yb)
                runing_val_loss += loss.item()
        avg_test_loss_per_epoch = runing_val_loss/ len(test_loader)
        test_loss.append(avg_test_loss_per_epoch)
        print(f"{epoch}/{epochs} train loss==> {avg_loss_per_epoch} || test loss ==> {avg_test_loss_per_epoch}")

        if best_epoch_loss  > avg_test_loss_per_epoch:
            best_epoch_loss = avg_test_loss_per_epoch
            torch.save(ann_model.state_dict(),"best_model.pt") # pt or pth

