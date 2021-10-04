from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import matplotlib.pyplot as plt

training_loss = list()
testing_loss = list()

df = pd.read_csv("./csv/heart.csv")
#X - features
#Y - outputs 

X = np.asarray(df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']])
Y = np.asarray(df['output'])
num_features = X.shape[1]
output_size = 1 #binary, so either 0 or 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# print(X_train[:20])
# print(Y_train[:20])

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
Y_train = torch.Tensor(Y_train).unsqueeze(-1)
Y_test = torch.Tensor(Y_test).unsqueeze(-1)
print(X_test[:20])
print(Y_test[:20])

class Network(nn.Module):
  def __init__(self, input_dim = num_features, output_dim=output_size):
    super().__init__()
    layers = []
    hidden_dim = 15
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    layers.append(nn.Sigmoid())

    self.net = nn.Sequential(*layers)
  def forward(self, x):
    return self.net(x)

#training and accuracy 
net = Network()
print(net)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())

num_epochs = 1001

for epoch in range(num_epochs):
    net.train()                             
    optimizer.zero_grad() #reset the gradient                  
    predictions = net(X_train) #pass data through the network             
    loss = criterion(predictions, Y_train) #calculating the loss
    loss.backward() #backpropagation                        
    optimizer.step() #updates the gradients                        

    if epoch%100 == 0:
        print("epoch", epoch, ":")
        print("training loss =", loss.item())

        with torch.no_grad():                                   
            net.eval()                                          
            test_predictions = net(X_test)                      
            test_loss = criterion(test_predictions, Y_test)
            test_predictions = torch.round(test_predictions)
            print("test loss =", test_loss.item())
            print("test accuracy =", accuracy_score(Y_test, test_predictions))

train_losses = training_loss
val_losses = testing_loss
print(len(train_losses))
print(len(val_losses))
plt.plot(train_losses, '-bx')
plt.plot(val_losses, '-rx')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Training', 'Validation'])
plt.title('Loss vs. No. of epochs')