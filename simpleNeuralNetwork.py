import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(
        self,
        neurouns_on_input_layer=784,
        neurouns_on_hidden_layer=30,
        neurouns_on_output_layer=10,
    ):

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

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_acurracy(self, prediction, y):
        return np.sum(prediction == y) / y.size

    def get_predictions(self, output_layer):
        return np.argmax(output_layer, 0)

    def forward_propagation(self, x):

        self.hidden_layer_without_activity = self.weights_input_layer_to_hidden_layer.T.dot(x.T) + self.bias_on_hidden_layer

        self.hidden_layer = self.reLU(self.hidden_layer_without_activity)

        self.output_layer_without_activity = self.weights_hidden_layer_to_output_layer.T.dot(
            self.hidden_layer
        ) + self.bias_on_output_layer

        self.output_layer = self.softmax(self.output_layer_without_activity)

        return self.output_layer

    def backpropagation(self, x, y):
        predicted_value = self.forward_propagation(x)

        m = y.size

        derivative_Z2 = 2*(predicted_value - self.expected_value(y))

        self.derivative_W2 = 1/m * np.dot(derivative_Z2, self.hidden_layer.T).T

        self.derivative_b2 = 1/m * np.sum(derivative_Z2, 1)

        derivative_Z1 = self.weights_hidden_layer_to_output_layer.dot(
            derivative_Z2
        ) * (self.reLU(self.hidden_layer_without_activity, derivative=True))

        self.derivative_b1 = 1/m * np.sum(derivative_Z1, 1)
        self.derivative_W1 = 1/m * np.dot(derivative_Z1, x).T

    def gradient_descent(self, learning_rate):

        self.weights_hidden_layer_to_output_layer -= learning_rate*self.derivative_W2
        self.weights_input_layer_to_hidden_layer -= learning_rate*self.derivative_W1

        self.bias_on_hidden_layer -= learning_rate*np.reshape(self.derivative_b1, (30, 1))
        self.bias_on_output_layer -= learning_rate*np.reshape(self.derivative_b2, (10, 1))

    def train(self, x, y, learning_rate, epochs):
        for epoch in range(epochs):
            self.backpropagation(x, y)
            self.gradient_descent(learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}")
                prediction = self.get_predictions(self.output_layer)
                print(f"Acurracy: {self.get_acurracy(prediction, y)}")

    def predict(self, x):
        output_layer = self.forward_propagation(x)
        predicted_value = self.get_predictions(output_layer)
        return predicted_value

    def test_prediction(self, index, test_data):
        current_image = test_data.iloc[index]
        current_image = np.matrix(current_image)
        prediction = self.predict(current_image)

        print(f"Prediction: {prediction}")
        print("Image:")
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()
