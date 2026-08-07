"""
complete implementation of ANN regression
"""
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class PreANN:
    """
    -> Scale the data
    -> convert np.array into tensor array
    -> Wrap-up X ,y ternsor for both train and test
    -> Convert into batches
    """

    def __init__(self,X,y):
        self.X = X
        self.y = y

    def pre_ann(self):
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        X_test_scaled = scalar.transform(X_test)

        X_train_ternsor = torch.tensor(X_train_scaled,dtype=torch.float32)
        X_test_ternsor = torch.tensor(X_test_scaled,dtype=torch.float32)
        y_train_ternsor = torch.tensor(y_train.values,dtype=torch.float32).view(-1,1)
        y_test_ternsor = torch.tensor(y_test.values,dtype=torch.float32).view(-1,1)

        # wrap-up function
        train_dataset = TensorDataset(X_train_ternsor,y_train_ternsor)
        test_dataset = TensorDataset(X_test_ternsor,y_test_ternsor)

        # convert into the batches
        train = DataLoader(train_dataset,batch_size=32,shuffle=True)
        test = DataLoader(test_dataset,batch_size=32,shuffle=True)

        return [train,test]



class ANN(nn.Module):
    def __init__(self,size):
        super(ANN,self).__init__()

        self.model = nn.Sequential(
            # here we define the architecture ANN network
            # this is hidden layer
            nn.Linear(size,8),
            nn.ReLU(),

            # this is hidden layer
            nn.Linear(8,6),
            nn.ReLU(),

            # this is hidden layer
            nn.Linear(6,6),
            nn.ReLU(),

            # this is output layer
            nn.Linear(6,1)
        )

    def forward(self,x):
            return self.model(x)



if __name__ == "__main__":
    df = pd.read_csv("./data/powerplant_data.csv")
    X = df.drop(columns=["PE"])
    y = df["PE"]


    pre_ann = PreANN(X,y)
    train,test = pre_ann.pre_ann()
    ann = ANN(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ann.parameters()) ## update the weight and bias during the traning

    epochs = 100

    train_loss = []
    test_loss = []
    best_score = float("inf")
    for epoch in range(epochs):

        ann.train()
        running_loss_train = 0 
        for xb,yb in train:
            optimizer.zero_grad() # it erase all the data about by how much weight and bias need to updated and start with zero
            output = ann(xb) # prediaction
            loss = criterion(output,yb) # loss 
            loss.backward()
            running_loss_train += loss.item() # add to the running loss 
            optimizer.step() # weight updation
        avg_train_loss = running_loss_train / len(train)
        train_loss.append(avg_train_loss)

        # for evaluation
        ann.eval()
        runing_val_loss = 0.0
        with torch.no_grad():
            for xb,xy in test:
                output = ann(xb)
                loss = criterion(output,xy)
                runing_val_loss += loss.item()

        avg_val_loss = runing_val_loss/len(test)
        test_loss.append(avg_val_loss)

        print(f"{epoch}/{epochs} train loss ==> {avg_train_loss} & test loss ==> {avg_val_loss}")

        if avg_val_loss < best_score:
            best_score = avg_val_loss
            
