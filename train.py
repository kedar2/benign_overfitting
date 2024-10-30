from dataset import DataModel
from model import TwoLayerNet
import torch
import numpy as np
from typing import Tuple

def train_till_convergence(input_dim: int,
                           num_samples: int,
                           hidden_dim: int,
                           corruption_rate: float,
                           signal_level: float, 
                           verbosity: int=-1,
                           stopping_threshold: float=1e-6,
                           num_trials: float=1) -> Tuple[float, float]:
    """
    Trains a model with a given set of hyperparameters
    and returns the train and generalization error after convergence to 0 loss.
    """
    data_model = DataModel(input_dim=input_dim, num_samples=num_samples, corruption_rate=corruption_rate, signal_level=signal_level)
    train_data = data_model.train_dataset
    train_X = train_data[0]
    train_y = train_data[1].float()

    model = TwoLayerNet(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=1/(hidden_dim * num_samples))

    train_error = np.inf
    generalization_error = np.inf
    epoch = 0
    while train_error > stopping_threshold:
        epoch += 1
        optimizer.zero_grad()
        outputs = model(train_X)

        # use hinge loss
        loss = torch.max(1 - outputs * (2 * train_y - 1), torch.zeros_like(outputs)).sum()
        loss.backward()
        optimizer.step()
        
        train_error = data_model.train_error(model)
        
        if verbosity == 1:
            print(f"Epoch {epoch}, Train error: {train_error}")
        if verbosity == 2:
            generalization_error = data_model.generalization_error(model)
            print(f"Epoch {epoch}, Train error: {train_error}, Generalization error: {generalization_error}")
        generalization_error = data_model.generalization_error(model)
    return train_error, generalization_error
