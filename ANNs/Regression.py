"""
Goal:
    Explore the relationship between the regression slope & the ability for my model to learn the relationship between x & y
"""
import numpy as np
from torch import nn, randn, Tensor, optim, zeros
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

def trainModel(
        model: nn.Sequential, 
        epoch: int, 
        x: Tensor, 
        y: Tensor, 
        optimizer: optim.SGD, 
        lossFunc: nn.MSELoss
    ) -> Tuple[float, float]:
    
    losses = zeros(epoch)
    for epochi in range(epoch):
        # forward pass
        yHat = model(x)

        # compute loss
        loss = lossFunc(yHat, y)
        losses[epochi] = loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    predictions: Tensor = model(x)
    finalLoss = (y - predictions)
    finalLoss = finalLoss.pow(2).mean()
    accuracy = 1 - finalLoss.detach()

    return finalLoss, accuracy

slopes = np.linspace(
    start= -5,
    stop= 5,
    num= 40
)

results = {
    "loss": [],
    "Accuracy": []
}

trialRuns = 50
epochs = 75

losses = zeros(slopes.size)
accuracies = zeros(slopes.size)


for mIndex in range(slopes.size):
    
    m = slopes[mIndex]
    model = createANN()
    opt = optim.SGD(model.parameters(), lr= 0.5)
    lossFunc = nn.MSELoss()

    x, y = createData(m, 50)
    loss, accuracy = trainModel(
        model,
        epochs,
        x,
        y,
        opt,
        lossFunc
    )

    print(loss, accuracy)
    losses[mIndex] = loss
    accuracies[mIndex] = accuracy

plt.plot(slopes, losses.detach(),'o',markerfacecolor='w',linewidth=.1)
plt.xlabel('Slope')
plt.ylabel('Loss')
plt.show()