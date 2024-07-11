#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Neural Network to learn XOR

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1 // XOR problem has 1 output
#define numTrainingsets 4

// Sigmoid activation function
double sigmoid(double x) { return 1 / (1 + exp(-x)); }

// Derivative of the sigmoid function
double dSigmoid(double x) { return x * (1 - x); }

// Initialize weights with random values
double init_weights()
{
    return ((double)rand()) / ((double)RAND_MAX);
}

// Shuffle the training set order
void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main(void)
{
    // Set the learning rate
    const double learningRate = 0.1;

    // Declare arrays for layers, biases, and weights
    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    // Define the training data (inputs and outputs for XOR)
    double training_inputs[numTrainingsets][numInputs] = {{0.0, 0.0},
                                                          {1.0, 0.0},
                                                          {0.0, 1.0},
                                                          {1.0, 1.0}};

    double training_outputs[numTrainingsets][numOutputs] = {{0.0},
                                                            {1.0},
                                                            {1.0},
                                                            {0.0}};

    // Initialize weights and biases with random values
    for (int i = 0; i < numInputs; i++)
    {
        for (int j = 0; j < numHiddenNodes; j++)
        {
            hiddenWeights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numHiddenNodes; i++)
    {
        hiddenLayerBias[i] = init_weights();
        for (int j = 0; j < numOutputs; j++)
        {
            outputWeights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numOutputs; i++)
    {
        outputLayerBias[i] = init_weights();
    }

    // Define the order of the training sets
    int trainingSetOrder[] = {0, 1, 2, 3};
    int numberOfEpochs = 10000; // Number of training iterations

    // Training loop
    for (int epoch = 0; epoch < numberOfEpochs; epoch++)
    {
        // Shuffle the training set order
        shuffle(trainingSetOrder, numTrainingsets);

        // Loop through each training set
        for (int x = 0; x < numTrainingsets; x++)
        {
            int i = trainingSetOrder[x];

            // Forward pass: Compute hidden layer activations
            for (int j = 0; j < numHiddenNodes; j++)
            {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++)
                {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            // Forward pass: Compute output layer activations
            for (int j = 0; j < numOutputs; j++)
            {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++)
                {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            // Print output log for each training set
            if (epoch % 1000 == 0) // Print every 1000 epochs
            {
                printf("Epoch: %d, Input: [%g, %g], Expected Output: [%g], Predicted Output: [%g]\n",
                       epoch, training_inputs[i][0], training_inputs[i][1],
                       training_outputs[i][0], outputLayer[0]);
            }

            // Backpropagation: Compute output layer error
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++)
            {
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            // Backpropagation: Compute hidden layer error
            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++)
            {
                double error = 0.0;
                for (int k = 0; k < numOutputs; k++)
                {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            // Update weights and biases for the output layer
            for (int j = 0; j < numOutputs; j++)
            {
                outputLayerBias[j] += deltaOutput[j] * learningRate;
                for (int k = 0; k < numHiddenNodes; k++)
                {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;
                }
            }

            // Update weights and biases for the hidden layer
            for (int j = 0; j < numHiddenNodes; j++)
            {
                hiddenLayerBias[j] += deltaHidden[j] * learningRate;
                for (int k = 0; k < numInputs; k++)
                {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * learningRate;
                }
            }
        }
    }

    // Print final outputs after training
    printf("\nFinal Outputs after Training:\n");
    for (int i = 0; i < numTrainingsets; i++)
    {
        // Forward pass to compute final outputs
        for (int j = 0; j < numHiddenNodes; j++)
        {
            double activation = hiddenLayerBias[j];
            for (int k = 0; k < numInputs; k++)
            {
                activation += training_inputs[i][k] * hiddenWeights[k][j];
            }
            hiddenLayer[j] = sigmoid(activation);
        }

        for (int j = 0; j < numOutputs; j++)
        {
            double activation = outputLayerBias[j];
            for (int k = 0; k < numHiddenNodes; k++)
            {
                activation += hiddenLayer[k] * outputWeights[k][j];
            }
            outputLayer[j] = sigmoid(activation);
        }

        printf("Input: [%g, %g], Expected Output: [%g], Predicted Output: [%g]\n",
               training_inputs[i][0], training_inputs[i][1],
               training_outputs[i][0], outputLayer[0]);
    }

    return 0;
}
