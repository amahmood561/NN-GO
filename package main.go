package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// NeuralNetwork represents the neural network structure
type NeuralNetwork struct {
	inputNeurons     int
	hiddenNeurons    int
	outputNeurons    int
	learningRate     float64
	weightsInputHidden [][]float64
	weightsHiddenOutput [][]float64
}

// NewNeuralNetwork creates a new neural network
func NewNeuralNetwork(inputNeurons, hiddenNeurons, outputNeurons int, learningRate float64) *NeuralNetwork {
	nn := &NeuralNetwork{
		inputNeurons:     inputNeurons,
		hiddenNeurons:    hiddenNeurons,
		outputNeurons:    outputNeurons,
		learningRate:     learningRate,
		weightsInputHidden: make([][]float64, inputNeurons),
		weightsHiddenOutput: make([][]float64, hiddenNeurons),
	}

	// Initialize weights randomly
	rand.Seed(time.Now().UnixNano())
	for i := range nn.weightsInputHidden {
		nn.weightsInputHidden[i] = make([]float64, hiddenNeurons)
		for j := range nn.weightsInputHidden[i] {
			nn.weightsInputHidden[i][j] = rand.Float64()
		}
	}
	for i := range nn.weightsHiddenOutput {
		nn.weightsHiddenOutput[i] = make([]float64, outputNeurons)
		for j := range nn.weightsHiddenOutput[i] {
			nn.weightsHiddenOutput[i][j] = rand.Float64()
		}
	}

	return nn
}

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Sigmoid derivative
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Train the neural network with given input and target output
func (nn *NeuralNetwork) Train(inputs, targets []float64) {
	// Forward pass
	hiddenInputs := make([]float64, nn.hiddenNeurons)
	for i := 0; i < nn.hiddenNeurons; i++ {
		for j := 0; j < nn.inputNeurons; j++ {
			hiddenInputs[i] += inputs[j] * nn.weightsInputHidden[j][i]
		}
		hiddenInputs[i] = sigmoid(hiddenInputs[i])
	}

	finalOutputs := make([]float64, nn.outputNeurons)
	for i := 0; i < nn.outputNeurons; i++ {
		for j := 0; j < nn.hiddenNeurons; j++ {
			finalOutputs[i] += hiddenInputs[j] * nn.weightsHiddenOutput[j][i]
		}
		finalOutputs[i] = sigmoid(finalOutputs[i])
	}

	// Calculate output errors
	outputErrors := make([]float64, nn.outputNeurons)
	for i := 0; i < nn.outputNeurons; i++ {
		outputErrors[i] = targets[i] - finalOutputs[i]
	}

	// Backpropagate errors to hidden layer
	hiddenErrors := make([]float64, nn.hiddenNeurons)
	for i := 0; i < nn.hiddenNeurons; i++ {
		for j := 0; j < nn.outputNeurons; j++ {
			hiddenErrors[i] += outputErrors[j] * nn.weightsHiddenOutput[i][j]
		}
	}

	// Update weights for hidden-output layer
	for i := 0; i < nn.hiddenNeurons; i++ {
		for j := 0; j < nn.outputNeurons; j++ {
			nn.weightsHiddenOutput[i][j] += nn.learningRate * outputErrors[j] * sigmoidDerivative(finalOutputs[j]) * hiddenInputs[i]
		}
	}

	// Update weights for input-hidden layer
	for i := 0; i < nn.inputNeurons; i++ {
		for j := 0; j < nn.hiddenNeurons; j++ {
			nn.weightsInputHidden[i][j] += nn.learningRate * hiddenErrors[j] * sigmoidDerivative(hiddenInputs[j]) * inputs[i]
		}
	}
}

// Predict outputs for given inputs
func (nn *NeuralNetwork) Predict(inputs []float64) []float64 {
	// Forward pass
	hiddenInputs := make([]float64, nn.hiddenNeurons)
	for i := 0; i < nn.hiddenNeurons; i++ {
		for j := 0; j < nn.inputNeurons; j++ {
			hiddenInputs[i] += inputs[j] * nn.weightsInputHidden[j][i]
		}
		hiddenInputs[i] = sigmoid(hiddenInputs[i])
	}

	finalOutputs := make([]float64, nn.outputNeurons)
	for i := 0; i < nn.outputNeurons; i++ {
		for j := 0; j < nn.hiddenNeurons; j++ {
			finalOutputs[i] += hiddenInputs[j] * nn.weightsHiddenOutput[j][i]
		}
		finalOutputs[i] = sigmoid(finalOutputs[i])
	}

	return finalOutputs
}

func main() {
	// Create a neural network
	nn := NewNeuralNetwork(2, 2, 1, 0.1)

	// Training data for XOR problem
	trainingInputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	trainingTargets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Train the network
	for i := 0; i < 10000; i++ {
		index := rand.Intn(4)
		nn.Train(trainingInputs[index], trainingTargets[index])
	}

	// Test the network
	for _, inputs := range trainingInputs {
		outputs := nn.Predict(inputs)
		fmt.Printf("Inputs: %v, Output: %v\n", inputs, outputs)
	}
}
