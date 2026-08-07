import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def pre_train_ann(X, y):
    try:
        l_enc = LabelEncoder()
        y = l_enc.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        X_test_scaled = scalar.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_tenrsor = TensorDataset(X_train_tensor, y_train_tensor)
        test_tensor = TensorDataset(X_test_tensor, y_test_tensor)

        train = DataLoader(train_tenrsor, batch_size=32, shuffle=True)
        test = DataLoader(test_tensor, batch_size=32,shuffle=True)
        return [train, test]
    except Exception as e:
        print(e)




class ANNClassifier(nn.Module):
    def __init__(self,size):
        super(ANNClassifier,self).__init__()
        self.model = nn.Sequential(

            # creating the hidden layers
            nn.Linear(size,64),
            nn.ReLU(),

            nn.Linear(64,64),
            nn.ReLU(),

            nn.Linear(64,7)
            # if we are using the crossEntropyloss as const function we don't need to use the softmax activation function explicitly
        )

    def forward(self,x):
        return self.model(x)



if __name__ == "__main__":
    df = pd.read_csv("data/DateFruit_Dataset.csv")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    train,test = pre_train_ann(X,y)

    
    epochs = 100
    train_loss = []
    best_loss = float("inf")
    criterion = nn.CrossEntropyLoss()
    model = ANNClassifier(X.shape[1])
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        tr_loss = 0.0
        model.train()

        for xb,yb in train:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output,yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() 

        avg_train_loss = tr_loss/len(train)
        train_loss.append(avg_train_loss)

        # print(f"{epoch}/{epochs} train loss == {avg_train_loss}")

    total_correct_pred = 0
    total = 0
    for xb,yb in test:
        output = model(xb) # ouput will the list of probabilities[]
        # new get maximum value and its index
        _,prediction = torch.max(output,1)
        total_correct_pred += (prediction == yb).sum().item()
        total += yb.size(0)

    print(f"{total_correct_pred} out of {total}")

