## Building a Neural Network from Scratch

### Data Loading and Preprocessing:

- The script begins by loading a CSV dataset of handwritten digits from the **MNIST dataset**.
- The dataset is converted to a numpy array and shuffled.
- It's then divided into training and development sets.
- The pixel values of the images are normalized to a range between 0 and 1.

### Model Parameters Initialization:

- The `init_params()` function initializes the weights and biases for the two layers of the neural network.
- Weights are initialized to small random values.
- Biases are also initialized to small random values.

### Activation Functions:

- The script defines two activation functions for the neural network:
  1. **ReLU** (Rectified Linear Unit) for the hidden layer.
  2. **Softmax** for the output layer.

### Forward Propagation:

- The `forward_prop()` function details the forward propagation step of the neural network, determining the predicted outputs.

### Backward Propagation and Parameter Updates:

- The script encompasses the backpropagation step, where the gradients of the loss function concerning the model's parameters are determined.
- Subsequently, the parameters are updated in the direction that minimally amplifies the loss function.

### Training the Model:

- The `gradient_descent()` function trains the model using the training data. 
- The process includes applying forward propagation, backpropagation, and parameter updates iteratively for a set number of cycles.

### Model Evaluation:

- The code encompasses functions to:
  1. Make predictions with the trained model (`make_predictions()`).
  2. Evaluate the accuracy of these predictions (`get_accuracy()`).

### Prediction Visualization:

- The `test_prediction()` function visualizes a prediction for an individual image.
- It exhibits the image and prints both the model's predicted digit and the true label.
