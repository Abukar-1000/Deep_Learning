import pandas as pd
from torch import Tensor, nn, optim, mean, zeros
import numpy as np
import matplotlib.pyplot as plt

class IrisModel(nn.Module):

    def __init__(self, units: int, layers: int, inputFeatures: int, outputFeatures: int) -> None:
        super().__init__()
        self.modelLayers = nn.ModuleDict()
        self.units = units
        self.layers = layers
        self.inputFeatures = inputFeatures
        self.outputFeatures = outputFeatures
        self.__createModel()

    def __createModel(self) -> None:
        self.modelLayers["input"] = nn.Linear(self.inputFeatures, self.units)

        for x in range(self.layers):
            self.modelLayers[f"layer{x}"] = nn.Linear(self.units, self.units)
        
        self.modelLayers["output"] = nn.Linear(self.units, self.outputFeatures)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.relu(self.modelLayers["input"](x))

        for layer in range(self.layers):
            x = nn.functional.relu(self.modelLayers[f"layer{layer}"](x))

        x = self.modelLayers["output"](x)

        return x

class SepalDataEnum:
    setosa: int = 0
    versiColor: int = 1
    virginica: int = 2

data = pd.read_csv("datasets/iris.csv")
trainData = data.sample(frac= 0.75, random_state= 150).reset_index(drop=True)
testData = data.drop(trainData.index).reset_index(drop=True)

trainSpeciesLabels = trainData[['sepal_length', 'sepal_width', 'petal_length', 'species']]
testSpeciesLabels = testData[['sepal_length', 'sepal_width', 'petal_length', 'species']]

trainData = Tensor(trainData[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values).float()
testData = Tensor(testData[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values).float()

trainSpecies = zeros(len(trainData)).long()
trainSpecies[trainSpeciesLabels['species'] == 'Iris-setosa'] = SepalDataEnum.setosa
trainSpecies[trainSpeciesLabels['species'] == 'Iris-versicolor'] = SepalDataEnum.versiColor
trainSpecies[trainSpeciesLabels['species'] == 'Iris-virginica'] = SepalDataEnum.virginica

testSpecies = zeros(len(testData)).long()
testSpecies[testSpeciesLabels['species'] == 'Iris-setosa'] = SepalDataEnum.setosa
testSpecies[testSpeciesLabels['species'] == 'Iris-versicolor'] = SepalDataEnum.versiColor
testSpecies[testSpeciesLabels['species'] == 'Iris-virginica'] = SepalDataEnum.virginica

"""
model = nn.Sequential(
    nn.Linear(4, 104),
    nn.functional.relu(),
    nn.Linear(104, 104),
    nn.functional.relu(),
    nn.Linear(104, 3)
)

"""

model = IrisModel(104, 4, 4, 3)

optimizer = optim.SGD(model.parameters(), lr=0.1)
lossFunc = nn.CrossEntropyLoss()
epoch = 1000
losses = np.zeros(epoch)

for epochi in range(epoch):
    # forward pass
    yhat = model(trainData)

    # compute loss
    loss = lossFunc(yhat, trainSpecies)
    losses[epochi] = loss

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# test model
predictions = model(testData)
accuracy = 100 * mean((predictions.argmax(dim=1) == testSpecies).float())
print(accuracy.item())

plt.plot(losses,'o',markerfacecolor='w',linewidth=.1)
plt.show()