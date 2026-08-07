import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split




class RNN(nn.Module):

    def __init__(self,input_size,hidden_size=128,num_layers=1):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define RNN layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # fully connected layers
        self.fc = nn.Linear(hidden_size,1)

    def forward(self,x):
        # optional to define the hidden state
        out,_= self.rnn(x)
        return self.fc(out[:,-1,:]) #all batches,last sequence len and all hidden size 
    

if __name__ == "__main__":
    df = pd.read_csv("../data/IMDB_dataset.csv")
    df.drop_duplicates(inplace=True)

    # text preprocessing
    df["review"] = df["review"].str.lower()

    # removing the urls
    df["review"] = df["review"].apply(lambda x: re.sub(r"http\S+", "", x))

    # remove the punctuation
    df["review"] = df["review"].apply(lambda x: re.sub(r"[^A-Za-z0-9]\s", " ", x))

    # remove html tag
    df["review"] = df["review"].apply(lambda x: re.sub(r"<.*?>", "", x))

    df["review"] = df["review"].apply(lambda x: word_tokenize(x))

    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    X = df.drop(columns=["sentiment"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    encoder = TfidfVectorizer(max_features=4000)
    
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    train_tensor = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )

    test_tensor = TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
    )   

    train_loader = DataLoader(train_tensor,batch_size=70,shuffle=True)
    test_tensor = DataLoader(test_tensor,batch_size=70,shuffle=True)


    model = RNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss() # binary cross entropy
    optimizer = optimizer.Adam(model.parameters())


    epochs = 10

    for epoch in range(epochs):
        model.train()

        for Xb,yb in train_loader:
            optimizer.zero_grad()
            Xb = Xb.unsqeeze(1) # add sigleton direction
            output = model(Xb) # (batch_size,1)

            output = torch.sigmoid(output.squeez()) # (batch_size,) 
            loss = criterion(output,yb)
            loss.backward()
            optimizer.step()

        print(f"{epoch}/{epochs} train loss = {loss}/{len(train_loader)}")


