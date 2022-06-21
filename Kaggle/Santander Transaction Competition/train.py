import torch
from sklearn import metrics  # We need ROC instead of just Accuracy
from tqdm import tqdm  # For progress bar
import torch.nn as nn
import torch.optim as optim
from utils import get_predictions
from dataset import get_data
from torch.utils.data import DataLoader
import torch.nn.functional as F


class NN(
    nn.Module):  # for a clearer and more concise training loop. We subclass nn.Module (which itself is a class and
    # able to keep track of state). We want to create a class that holds our weights, bias, and method for the
    # forward step. nn.Module has a number of attributes and methods (such as .parameters() and .zero_grad()
    def __init__(self, input_size, hidden_dim):
        super(NN, self).__init__()

        # The idea here is that, since there are no correlations,
        # we can make every feature its own example...
        # so each feature(one value) goes through a linear layer
        # which is mapped to some higher dim(hidden_dim)...
        # each feature is going through an embedding to create some
        # better representation of that single floating value...
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(input_size * hidden_dim, 1)

        # self.net = nn.Sequential(
        #     nn.BatchNorm1d(input_size),
        #     nn.Linear(input_size, 50),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(50, 1),
        # )

    # def forward(self, x):
    #     return torch.sigmoid(self.net(x)).view(-1)

    def forward(self, x):
        BatchSize = x.shape[0]
        x = self.bn(x)
        # Current shape of x : (BatchSize, 200)  16 , 200  eg. Lets say for BatchSize of 16 and hidden_dim of 5

        x = x.view(-1, 1)  # (BatchSize*200, 1)  16*200->(3200 , 1)

        # self.fc1 will spit 3200 * 5 --> 16000

        x = F.relu(self.fc1(x)).reshape(BatchSize, -1)  # (BatchSize, input_size*hidden_dim) 16 , 200*5

        return torch.sigmoid(self.fc2(x)).view(-1)  # Flatten


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = NN(input_size=200, hidden_dim=16).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

loss_fn = nn.BCELoss()

train_ds, val_ds, test_ds, test_ids = get_data()

train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

X, y = next(iter(train_loader))
# print(X.shape)


for epoch in range(15):
    probabilities, true = get_predictions(val_loader, model, device=DEVICE)
    print(f'VALIDATION ROC: {metrics.roc_auc_score(true, probabilities)}')
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward
        scores = model(data)
        # print(scores.shape)

        loss = loss_fn(scores, targets)
        # print(loss)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
