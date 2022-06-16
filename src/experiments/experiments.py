"""
Provides functions for running experiments, with wandb tracking.

Authors:
    Rahul Yedida <rahul@ryedida.me>
"""
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb

from raise_utils.metrics import ClassificationMetrics
from nebulgym.decorators.torch_decorators import accelerate_model


@accelerate_model()
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(19, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def run_experiment(name: str, train_ds: Dataset, test_ds: Dataset, model: nn.Module):
    """
    This shows an alternate form of docstring for functions.

    Runs one experiment, given a Data instance.

    :param {str} name - The name of the experiment.
    :param {Dataset} train_ds - The train dataset
    :param {Dataset} test_ds - The test dataset
    :param {Module} model - The model
    """
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=True)
    opt = optim.Adam(model.parameters())
    loss_func = F.binary_cross_entropy

    # wandb.init(project='dl4se-demo-torch')

    # 30 epochs
    for epoch in range(30):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred.reshape((pred.shape[0])), yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

    # Get the results.
    metrics_list = ['f1', 'd2h', 'pd', 'pf', 'prec']

    preds = []
    y = []
    for xb, yb in test_dl:
        preds.extend(model(xb).detach().cpu().numpy().argmax(axis=1))
        y.extend(yb.detach().cpu().numpy())

    m = ClassificationMetrics(y, preds)
    m.add_metrics(metrics_list)
    results = m.get_metrics()

    print(results)

    #wandb.log(dict(zip(metrics_list, results)))

    return results
