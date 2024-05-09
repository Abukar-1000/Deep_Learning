"""
Goal:
    Explore the relationship between the regression slope & the ability for my model to learn the relationship between x & y
"""
import numpy as np
from torch import nn, randn, Tensor
import matplotlib.pyplot as plt
from typing import Tuple

def createANN() -> nn.Sequential:
    ANNModel = nn.Sequential(
        nn.Linear(1,1),
        nn.ReLU(),
        nn.Linear(1,1)
    )

    return ANNModel

def createData(m: float, N:int) -> Tuple[Tensor, Tensor]:
    x = randn(N,1)
    y = m*x + randn(N,1)/2

    return x,y

def trainModel(model: nn.Sequential, epoch: int, x: Tensor):
    
    for epochi in range(epoch):
        # forward pass
        yHat = model(x)

        # compute loss

        # backprop

slopes = np.linspace(
    start= -2,
    stop= 2,
    num= 21
)

results = {
    "loss": [],
    "Accuracy": []
}

