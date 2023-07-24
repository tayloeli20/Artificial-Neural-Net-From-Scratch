# MNIST Digit Recognition with Neural Network

## Description

This project implements a simple neural network to recognize handwritten digits from the MNIST dataset. The goal is to achieve high accuracy in digit classification using forward and backward propagation.

## Dependencies

The project is implemented in C and requires the following dependencies:
- Standard C libraries
- Math library (for exponential function)

## Data Description

The project uses the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0 to 9). The dataset is split into training and testing sets. Each image is preprocessed to form a 784-dimensional input vector.

## Methodology and Algorithms

The neural network consists of one hidden layer with one unit and a softmax output layer. The ReLU activation function is used for the hidden layer, and the softmax activation function is used for the output layer.

### Forward Propagation
The input data is fed through the network to compute the output probabilities for each class.

### Backward Propagation
The gradients of the parameters are calculated to minimize the cross-entropy loss.

### Gradient Descent
The model's parameters are updated using gradient descent with a specified learning rate.

## Code Explanation

The code is organized into the following main functions:

- `init_params`: Initializes the parameters (weights and biases) of the neural network.
- `forward_prop`: Performs forward propagation for one step using the current parameters.
- `backward_prop`: Performs backward propagation for one step to compute gradients.
- `update_params`: Updates the parameters using gradient descent.
- `get_prediction`: Calculates the predicted class index from the output probabilities.
- `get_accuracy`: Calculates the accuracy of the predictions.

## Usage

To use the project, follow these steps:

1. Install the necessary dependencies and compile the C code.
2. Load the MNIST dataset and preprocess the images to create the input data.
3. Initialize the neural network parameters using `init_params`.
4. Perform gradient descent using `gradient_descent` with appropriate hyperparameters (learning rate, iterations).
5. Evaluate the model's performance and accuracy using `get_accuracy`.

## Results

The model's performance is evaluated throughout the training process. The accuracy and loss metrics are monitored to assess the convergence and effectiveness of the neural network.

## Limitations and Future Work

- The current implementation uses a simple neural network with one hidden unit, which may limit its ability to achieve high accuracy on complex datasets.
- Future work may involve experimenting with different network architectures, activation functions, and optimization techniques to improve performance.

## Contributing

This project is open to contributions. Feel free to submit bug reports, feature requests, or pull requests.

## License

This project is licensed under the MIT License.

## Contact Information

For any questions or feedback, please keep them to youself
