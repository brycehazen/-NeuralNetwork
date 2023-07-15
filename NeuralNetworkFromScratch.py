import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load data
digit_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
digit_data = np.array(digit_data)
np.random.shuffle(digit_data)

# Preprocess and split data into features and labels
features = digit_data[:, 1:].T / 255.0
labels = digit_data[:, 0].T
one_hot_labels = one_hot_encode(labels)
train_features, dev_features = features[:, 1000:], features[:, :1000]
train_labels, dev_labels = one_hot_labels[:, 1000:], one_hot_labels[:, :1000]

def one_hot_encode(labels):
    one_hot_encoded_labels = np.zeros((int(labels.max()) + 1, labels.size))
    one_hot_encoded_labels[labels.astype(int), np.arange(labels.size)] = 1
    return one_hot_encoded_labels

def initialize_parameters():
    layer_1_weights = np.random.rand(10, 784) - 0.5
    layer_1_bias = np.random.rand(10, 1) - 0.5
    layer_2_weights = np.random.rand(10, 10) - 0.5
    layer_2_bias = np.random.rand(10, 1) - 0.5
    return layer_1_weights, layer_1_bias, layer_2_weights, layer_2_bias

def ReLU(activations):
    return np.maximum(activations, 0)

def softmax(activations):
    exps = np.exp(activations - np.max(activations, axis=0))  # Subtract max for numerical stability
    return exps / np.sum(exps, axis=0)

def forward_propagation(layer_1_weights, layer_1_bias, layer_2_weights, layer_2_bias, features):
    layer_1_activations = layer_1_weights.dot(features) + layer_1_bias
    layer_1_output = ReLU(layer_1_activations)
    layer_2_activations = layer_2_weights.dot(layer_1_output) + layer_2_bias
    layer_2_output = softmax(layer_2_activations)
    return layer_1_activations, layer_1_output, layer_2_activations, layer_2_output

def ReLU_derivative(activations):
    return activations > 0

def backward_propagation(layer_1_activations, layer_1_output, layer_2_activations, layer_2_output, layer_1_weights, layer_2_weights, features, labels):
    m = labels.shape[1]
    layer_2_error = layer_2_output - labels
    layer_2_weight_gradient = layer_2_error.dot(layer_1_output.T) / m
    layer_2_bias_gradient = np.sum(layer_2_error, axis=1, keepdims=True) / m
    layer_1_error = layer_2_weights.T.dot(layer_2_error) * ReLU_derivative(layer_1_activations)
    layer_1_weight_gradient = layer_1_error.dot(features.T) / m
    layer_1_bias_gradient = np.sum(layer_1_error, axis=1, keepdims=True) / m
    return layer_1_weight_gradient, layer_1_bias_gradient, layer_2_weight_gradient, layer_2_bias_gradient

def update_parameters(layer_1_weights, layer_1_bias, layer_2_weights, layer_2_bias, layer_1_weight_gradient, layer_1_bias_gradient, layer_2_weight_gradient, layer_2_bias_gradient, learning_rate):
    layer_1_weights -= learning_rate * layer_1_weight_gradient
    layer_1_bias -= learning_rate * layer_1_bias_gradient    
    layer_2_weights -= learning_rate * layer_2_weight_gradient
    layer_2_bias -= learning_rate * layer_2_bias_gradient
    return layer_1_weights, layer_1_bias, layer_2_weights, layer_2_bias

def gradient_descent(features, labels, learning_rate, iterations):
    layer_1_weights, layer_1_bias, layer_2_weights, layer_2_bias = initialize_parameters()
    for _ in range(iterations):
        layer_1_activations, layer_1_output, layer_2_activations, layer_2_output = forward_propagation(layer_1_weights, layer_1_bias, layer_2_weights, layer_2_bias, features)
        layer_1_weight_gradient, layer_1_bias_gradient, layer_2_weight_gradient, layer_2_bias_gradient = backward_propagation(layer_1_activations, layer_1_output, layer_2_activations, layer_2_output, layer_1_weights, layer_2_weights, features, labels)
        layer_1_weights, layer_1_bias, layer_2_weights, layer_2_bias = update_parameters(layer_1_weights, layer_1_bias, layer_2_weights, layer_2_bias, layer_1_weight_gradient, layer_1_bias_gradient, layer_2_weight_gradient, layer_2_bias_gradient, learning_rate)
    return layer_1_weights, layer_1_bias, layer_2_weights, layer_2_bias

gradient_descent(train_features, train_labels, 0.10, 500)
