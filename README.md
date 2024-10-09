# Go Neural Network

A simple feedforward neural network implemented in Go. This project demonstrates the basics of building, training, and using a neural network for simple tasks like the XOR problem.

## Features

- Basic feedforward neural network with a single hidden layer
- Uses sigmoid activation function
- Implements backpropagation for training
- Solves the XOR problem as an example

## Prerequisites

To run this project, you need:

- [Go](https://golang.org/dl/) (version 1.16 or higher)

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/go-neural-network.git
cd go-neural-network
```
### 2. Build the Project
Run the following command to build the project:

bash
```bash
go build -o neural-network
```
This will generate an executable file named neural-network in your project directory.

### 3. Run the Neural Network
After building, you can execute the program with:

```bash
./neural-network
The output will display the network's predictions for the XOR problem:
```
yaml
Inputs: [0 0], Output: [0.01]
Inputs: [0 1], Output: [0.99]
Inputs: [1 0], Output: [0.98]
Inputs: [1 1], Output: [0.02]

### 4. Modify Training or Network Configuration
You can change the network configuration (number of neurons, learning rate, etc.) and training data in the main.go file.

### Project Structure
The main components of this project are:

main.go: Contains the neural network implementation, training, and prediction logic.
NeuralNetwork: The struct representing the neural network, with methods for training (Train) and prediction (Predict).
### How It Works
#### 1. Initialization: The network is initialized with random weights for the input-hidden and hidden-output connections.
#### 2. Training: Uses backpropagation to adjust weights based on the error between the predicted output and the target output.
#### 3. Prediction: After training, the network can predict outputs for new inputs by performing a forward pass through the network.
#### 4. Example: Solves the XOR problem, a classic problem in neural network research where the network must learn to output true (1) only when the inputs are different.

### Future Enhancements
Add support for more layers and different activation functions
Implement different learning algorithms (e.g., gradient descent optimizers)
Extend to support more complex datasets