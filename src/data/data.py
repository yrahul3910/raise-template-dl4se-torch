"""
Provides data fetching and preprocessing functionalities. Some code taken from
https://github.com/yrahul3910/raise/blob/master/raise_utils/data/data.py

Authors:
    Rahul Yedida <rahul@ryedida.me>
"""
import torch
from torch.utils.data import Dataset
from nebulgym.decorators.torch_decorators import accelerate_dataset
import numpy as np
import pandas as pd
import os


def get_data(base_path: str, files: list):
    """
    Fetches data from a list of data files, and binarizes it. Returns a pair of
    PyTorch Dataset instances.

    :param {str} base_path - The base path to the data files.
    :param {list} files - A list of files to load.
    """
    paths = [os.path.join(base_path, file_name) for file_name in files]
    train_df = pd.concat([pd.read_csv(path)
                          for path in paths[:-1]], ignore_index=True)
    test_df = pd.read_csv(paths[-1])

    # Magic number 3 is the first numeric column index
    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]

    train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
    test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)

    X_train = train_df[train_df.columns[:-2]]
    y_train = train_df['bug'].astype('int')
    X_test = test_df[test_df.columns[:-2]]
    y_test = test_df['bug'].astype('int')

    return DefectPredictionDataset(X_train, y_train), DefectPredictionDataset(X_test, y_test)


@accelerate_dataset()
class DefectPredictionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.FloatTensor(np.array(y))

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)
