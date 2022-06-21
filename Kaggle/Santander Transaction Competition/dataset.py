import pandas as pd
import torch
from torch.utils.data import TensorDataset  # to create the dataset...represent your data as list of tensors.
# you can read the entire data in memory and it will automatically create a torch dataset for you...

from torch.utils.data.dataset import random_split
from math import ceil


def get_data():  # Load the training/testing data

    train_data = pd.read_csv('train.csv')
    y = train_data["target"]
    X = train_data.drop(['ID_code', 'target'], axis=1)

    # Now we convert them to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # We want the validation data as well
    ds = TensorDataset(X_tensor, y_tensor)
    train_ds, val_ds = random_split(ds, [int(0.8 * len(ds)), ceil(0.2 * len(ds))])

    # Doing the same for the test data
    test_data = pd.read_csv('test.csv')

    # For submission
    test_ids = test_data["ID_code"]

    X = test_data.drop(['ID_code'], axis=1)

    # Now we convert them to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)  # <----------------------- y value from above??

    test_ds = TensorDataset(X_tensor, y_tensor)

    return train_ds, val_ds, test_ds, test_ids
