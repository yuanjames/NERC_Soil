from concurrent.futures import process
from tkinter import Y
import pandas as pd
import numpy as np


def extraction_data(
    file: str = "Raw data for EDCs NSIS2 MA.xlsx",
    sheet_index_X: int = 2,
    sheet_index_y: int = 0,
    feature: list = [1, -1],
    task: str = "DEHP",
    clean: bool = True,
):
    """
    User could use this function to extract data (x and y) frome xlsx file. You need to specify the indices of sheets respectively for X and y.
    
    Args:
    file: str, the path  of input data
    feature: tuple, indexes for selecting features.
    tasks: str, select a specific task to do. Mutitasks is not supported.
    
    """
    X = pd.read_excel(file, sheet_name=sheet_index_X).iloc[:, feature[0] : feature[1]]
    y = pd.read_excel(file, sheet_name=sheet_index_y)[task]

    assert X.shape[0] == y.shape[0]
    print("Data amount from the orginal file:", X.shape[0])

    feat_names = X.columns.values

    if clean:

        return clean_data(X, y, feat_names)

    print("Data amount:", X.shape[0])
    return X, y, feat_names


def clean_data(
    X,
    y,
    feat_names,
    feature: str = "Elevation",
    to_drop=-9999.0,
    to_replace_y: list = ["<0.05"],
    value_y: list = [0.05],
):
    """
    User could call this function in extract_data function, this function is used to drop some undesirable data.
    
    Args:
    X, input.
    y, output.
    feature: which column is used to detect undesirable data.
    to_drop: if it is equal to the value, the row of data will be dropped.
    
    """
    index_drop = X[X[feature] == to_drop].index
    X.drop(index_drop, inplace=True)
    y.drop(index_drop, inplace=True)
    y.replace(to_replace_y, value_y, inplace=True)

    print("Data amount after clean:", X.shape[0])
    print("The number of data was dropped", len(index_drop))

    return X, y, feat_names
