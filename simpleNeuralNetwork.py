import numpy as np
import pandas as pd
from random import randrange


class NeuralNetwork:
    def __init__(
        self,
        learning_rate,
        neurouns_on_input_layer=785,
        neurouns_on_hidden_layer=30,
        neurouns_on_output_layer=10,
    ):
        self.learning_rate = learning_rate

        self.input_layer = np.zeros(neurouns_on_input_layer)
        self.hidden_layer = np.zeros(neurouns_on_hidden_layer)
        self.output_layer = np.zeros(neurouns_on_output_layer)

        self.weights_input_layer_to_hidden_layer = np.random.rand(
            neurouns_on_input_layer, neurouns_on_hidden_layer
        )
        self.weights_layer_to_output_layer = np.random.rand(
            neurouns_on_hidden_layer, neurouns_on_output_layer
        )

        self.bias_on_hidden_layer = np.full(
            (neurouns_on_hidden_layer, 1), randrange(1, 10)
        )

        self.bias_on_output_layer = np.full(
            (neurouns_on_output_layer, 1), randrange(1, 10)

        )

    def expected_value(self, x):
        expected_values = np.zeros((1, 10))
        expected_values[0][x] = 1
        return expected_values

    def reLU(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_of_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def loss_function(self, predicted_value, actual_value):
        return np.sum(np.subtract(predicted_value, actual_value)**2)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward_propagation(self, x):
        x = np.array(x)
        self.input_layer = x + self.input_layer

        self.hidden_layer_without_activity = np.dot(
            self.input_layer, self.weights_input_layer_to_hidden_layer
        )

        self.hidden_layer = self.sigmoid(
            self.hidden_layer_without_activity + self.bias_on_hidden_layer.T
        )

        self.output_layer_without_activity = np.dot(
            self.hidden_layer, self.weights_layer_to_output_layer
        )
        self.output_layer = self.sigmoid(
            self.output_layer_without_activity + self.bias_on_output_layer.T
        )
        return self.output_layer

    def backpropagation(self, x, y, mini_batch_size=500):
        x, y = np.array(x), np.array(y)
        epoch = 1
        x_samples = np.array_split(x, mini_batch_size)
        y_samples = np.array_split(y, mini_batch_size)
        vector_of_desired_changes = np.array([])
        for x, y in zip(x_samples, y_samples):
            epoch += 1
            predicted_value = self.forward_propagation(x)
            cost_value = self.loss_function(predicted_value.T, y)
            desired_change = np.subtract(cost_value, self.expected_value(y))
            vector_of_desired_changes = np.append(vector_of_desired_changes, desired_change)
        return vector_of_desired_changes

    def gradient_descent(self, x, y):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


if __name__ == "__main__":
    train_data = pd.read_csv("./datasets/MNIST/train.csv")
    test_data = pd.read_csv("./datasets/MNIST/test.csv")

    train_data = train_data.sample(frac=1, random_state=1).reset_index()

    X_train = train_data.drop("label", axis=1)
    y_train = train_data.label

    nl = NeuralNetwork(0.01)

    print(len(nl.backpropagation(X_train, y_train)))
