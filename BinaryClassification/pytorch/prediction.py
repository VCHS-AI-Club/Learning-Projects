import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


T = t.TypeVar("T")
LayerT = t.Union[nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Linear, nn.Sigmoid, None]


training_loss: list = list()
testing_loss: list = list()

df: pd.DataFrame = pd.read_csv("./csv/heart.csv")
# X - features
# Y - outputs

X: np.ndarray = np.asarray(
    df[
        [
            "age",
            "sex",
            "cp",
            "trtbps",
            "chol",
            "fbs",
            "restecg",
            "thalachh",
            "exng",
            "oldpeak",
            "slp",
            "caa",
            "thall",
        ]
    ]
)
Y: np.ndarray = np.asarray(df["output"])
num_features: tuple[int] = X.shape[1]
output_size: int = 1  # binary, so either 0 or 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# print(X_train[:20])
# print(Y_train[:20])

X_train: torch.Tensor = torch.Tensor(X_train)
X_test: torch.Tensor = torch.Tensor(X_test)
Y_train: torch.Tensor = torch.Tensor(Y_train).unsqueeze(-1)
Y_test: torch.Tensor = torch.Tensor(Y_test).unsqueeze(-1)
print(X_test[:20])
print(Y_test[:20])


class Network(nn.Module):
    """Wrapper class for Neural Network Module"""

    def __init__(
        self, input_dim: tuple[int] = num_features, output_dim: int = output_size
    ) -> None:
        super().__init__()
        layers: list[LayerT] = []
        hidden_dim: int = 15
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: T) -> nn.Sequential[T]:
        """Create a network from `x`"""
        return self.net(x)


# training and accuracy
net: Network = Network()
print(net)
criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
optimizer: optim.Adam = optim.Adam(net.parameters())

num_epochs: t.Final[int] = 1001  # set number of training cycles

for epoch in range(num_epochs):
    net.train()
    optimizer.zero_grad()  # reset the gradient
    predictions: Network = net(X_train)  # pass data through the network
    loss: nn.BCEWithLogitsLoss = criterion(predictions, Y_train)  # calculating the loss
    loss.backward()  # backpropagation
    optimizer.step()  # updates the gradients

    if epoch % 100 == 0:
        print("epoch", epoch, ":")
        print("training loss =", loss.item())

        with torch.no_grad():
            net.eval()
            test_predictions: Network = net(X_test)
            test_loss: nn.BCEWithLogitsLoss = criterion(test_predictions, Y_test)
            test_predictions: torch.Tensor = torch.round(test_predictions)
            # print("test loss =", test_loss.item())
            # print("test accuracy =", accuracy_score(Y_test, test_predictions))
            print(f"{test_loss.item()=}")
            print(f"{accuracy_score(Y_test, test_predictions)=}")


train_losses: list = training_loss
val_losses: list = testing_loss
print(len(train_losses))
print(len(val_losses))
plt.plot(train_losses, "-bx")
plt.plot(val_losses, "-rx")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["Training", "Validation"])
plt.title("Loss vs. No. of epochs")
