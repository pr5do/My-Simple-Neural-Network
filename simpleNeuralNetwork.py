import numpy as np
from random import randrange
import pandas as pd


def reLU(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def lossFunction(predictedValue, actualValue):
    """
        Receives two vectors as input and returns a floating number
    """
    return np.sum(np.subtract(predictedValue, actualValue)**2)


class NeuralNetwork():
    def __init__(self, neurounsOnInputLayer=784, neurounsOnHiddenLayer1=30, neurounsOnOutputLayer=10):
        self.inputLayer = np.zeros(neurounsOnInputLayer)
        self.weightsInputLayertoLayer1 = np.random.rand(neurounsOnInputLayer, neurounsOnHiddenLayer1)
        self.hiddenLayer1 = np.zeros(neurounsOnHiddenLayer1)
        self.biasOnHiddenLayer1 = np.random.rand(neurounsOnHiddenLayer1)
        self.weightsLayer1toOutputLayer = np.random.rand(neurounsOnHiddenLayer1, neurounsOnOutputLayer)
        self.biasOnOutputLayer = np.random.rand(neurounsOnOutputLayer)
        self.outputLayer = np.zeros(neurounsOnOutputLayer)

    def gradientDescendent(self, costValue, miniBatchSize=500):
        pass

    def train(self, XValues, YValues):
        pass


if __name__ == "__main__":
    train_data = pd.read_csv('./datasets/MNIST/train.csv')
    test_data = pd.read_csv('./datasets/MNIST/test.csv')

    np.random.shuffle(train_data)

    X_train = train_data.drop('label', axis=1)
    y_train = train_data.label

    nl = NeuralNetwork()
