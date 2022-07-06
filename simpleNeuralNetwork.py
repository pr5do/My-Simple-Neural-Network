import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(
        self,
        neurouns_on_input_layer=785,
        neurouns_on_hidden_layer=30,
        neurouns_on_output_layer=10,
    ):

        self.hidden_layer = np.zeros((neurouns_on_hidden_layer, 1))
        self.output_layer = np.zeros((neurouns_on_output_layer, 1))

        self.weights_input_layer_to_hidden_layer = np.random.rand(
             neurouns_on_input_layer, neurouns_on_hidden_layer
        )
        self.weights_hidden_layer_to_output_layer = np.random.rand(
            neurouns_on_hidden_layer, neurouns_on_output_layer
        )

        self.bias_on_hidden_layer = np.random.rand(neurouns_on_hidden_layer, 1)
        self.bias_on_output_layer = np.random.rand(neurouns_on_output_layer, 1)

    def expected_value(self, x):
        expected_value = np.zeros((10, 1))
        expected_value[x] = 1
        # expected_value = expected_value.T
        return expected_value

    def reLU(self, x, derivative=False):
        if derivative:
            return x > 0
        else:
            return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        if derivative:
            derivative = False
            return self.sigmoid(x)*(1-self.sigmoid(x))
        return 1 / (1 + np.exp(-x))

    def loss_function(self, predicted_value, actual_value, derivative=False):
        if derivative:
            return np.sum(2*np.subtract(predicted_value, actual_value))
        else:
            return np.sum(np.power(np.subtract(predicted_value, actual_value), 2))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward_propagation(self, x):

        self.hidden_layer_without_activity = np.dot(self.weights_input_layer_to_hidden_layer.T, x.T) + self.bias_on_hidden_layer

        self.hidden_layer = self.sigmoid(self.hidden_layer_without_activity)

        self.output_layer_without_activity = np.dot(self.weights_hidden_layer_to_output_layer.T, self.hidden_layer) + self.bias_on_output_layer

        self.output_layer = self.sigmoid(self.output_layer_without_activity)

        return self.output_layer

    def backpropagation(self, x, y):
        predicted_value = self.forward_propagation(x)
        cost_value = self.loss_function(predicted_value.T, self.expected_value(y))
        print(f"Cost Value: {cost_value}")

        m = y.size

        derivative_Z2 = predicted_value - self.expected_value(y)
        derivative_W2 = 1/m * np.dot(derivative_Z2, self.hidden_layer.T).T
        assert derivative_W2.shape == self.weights_hidden_layer_to_output_layer.shape
        derivative_b2 = 1/m * np.sum(derivative_Z2, axis=1, keepdims=True)
        assert derivative_b2.shape == self.bias_on_output_layer.shape
        derivative_Z1 = np.dot(self.weights_hidden_layer_to_output_layer, derivative_Z2) * (1 - np.power(self.hidden_layer, 2))
        derivative_b1 = 1/m * np.sum(derivative_Z1, axis=1, keepdims=True)
        assert derivative_b1.shape == self.bias_on_hidden_layer.shape
        derivative_W1 = 1/m * np.dot(derivative_Z1, x).T
        assert derivative_W1.shape == self.weights_input_layer_to_hidden_layer.shape

        return derivative_W2, derivative_b2, derivative_W1, derivative_b1

    def gradient_descent(
            self,
            derivative_W2,
            derivative_b2,
            derivative_W1,
            derivative_b1,
            learning_rate):

        self.weights_hidden_layer_to_output_layer = self.weights_hidden_layer_to_output_layer -  learning_rate*derivative_W2
        self.weights_input_layer_to_hidden_layer = self.weights_input_layer_to_hidden_layer -  learning_rate*derivative_W1

        self.bias_on_hidden_layer = self.bias_on_hidden_layer - learning_rate*derivative_b1
        self.bias_on_output_layer = self.bias_on_output_layer - learning_rate*derivative_b2


    def train(self, x, y, learning_rate, mini_batch_size=500):
        x, y = np.array(x), np.array(y)
        x_samples = np.array_split(x, mini_batch_size)
        y_samples = np.array_split(y, mini_batch_size)
        epoch = 1
        for x_sample, y_sample in zip(x_samples, y_samples):
            print(f"Epoch: {epoch}")
            epoch += 1
            for x, y in zip(x_sample, y_sample):
                x = np.reshape(x, (1, 785))
                derivative_W2, derivative_b2, derivative_W1, derivative_b1 = self.backpropagation(x, y)
                self.gradient_descent(derivative_W2, derivative_b2, derivative_W1, derivative_b1, learning_rate)

    def predict(self, x):
        predicted_value = self.forward_propagation(x)

        return predicted_value


if __name__ == "__main__":
    train_data = pd.read_csv("../datasets/MNIST/train.csv")
    test_data = pd.read_csv("../datasets/MNIST/test.csv")

    train_data = train_data.sample(frac=1, random_state=1).reset_index()

    X_train = train_data.drop("label", axis=1)
    y_train = train_data.label

    nl = NeuralNetwork()

    nl.train(X_train, y_train, 0.01)

    prediction = nl.predict(test_data)


