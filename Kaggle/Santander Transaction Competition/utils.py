import pandas as pd
import numpy as np
import torch


def get_predictions(loader, model, device):
    model.eval()  # for the batchnorm....

    saved_preds = []

    true_labels = []

    with torch.no_grad():  # temporarily set all the requires_grad flag to false...
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            scores = model(X)

            saved_preds += scores.tolist()

            true_labels += y.tolist()

    model.train()
    return saved_preds, true_labels
