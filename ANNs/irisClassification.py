import pandas as pd
from torch import Tensor, nn, optim, mean, zeros
import numpy as np

class SepalDataEnum:
    setosa = 1.0
    versiColor = 2.0
    virginica = 3.0

data = pd.read_csv("datasets/iris.csv")
trainData = data.sample(frac= 0.75, random_state= 150)
testData = data.drop(trainData.index)

trainSpeciesLabels = trainData[['sepal_length', 'sepal_width', 'petal_length', 'species']]
testSpeciesLabels = testData[['sepal_length', 'sepal_width', 'petal_length', 'species']]

# print(trainSpeciesLabels)
trainData = Tensor(trainData[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values).float()
testData = Tensor(testData[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values).float()

# correct labels
trainSpecies = zeros(len(trainData)).float()
print(trainData)
trainSpecies[trainSpeciesLabels['species'] == 'Iris-setosa'] = SepalDataEnum.setosa
trainSpecies[trainSpeciesLabels['species'] == 'Iris-versicolor'] = SepalDataEnum.versiColor
trainSpecies[trainSpeciesLabels['species'] == 'Iris-virginica'] = SepalDataEnum.virginica

testSpecies = zeros((testData.shape[0], 4)).float()
testSpecies[testData['species'] == 'Iris-setosa'] = SepalDataEnum.setosa
testSpecies[testData['species'] == 'Iris-versicolor'] = SepalDataEnum.versiColor
testSpecies[testData['species'] == 'Iris-virginica'] = SepalDataEnum.virginica




model = nn.Sequential(
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 4)
)



optimizer = optim.SGD(model.parameters(), lr= 0.5)
lossFunc = nn.BCEWithLogitsLoss()
epoch = 500
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
accuracy = 100 * mean((predictions == testSpecies).float())
print(accuracy)