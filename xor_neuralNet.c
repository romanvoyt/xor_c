#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Neural Network to learn XOR

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1 // Fixed the output dimension to match XOR problem
#define numTrainingsets 4

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

double init_weights()
{
    return ((double)rand()) / ((double)RAND_MAX);
}

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
    const double learningRate = 0.1;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingsets][numInputs] = {{0.0, 0.0},
                                                          {1.0, 0.0},
                                                          {0.0, 1.0},
                                                          {1.0, 1.0}};

    double training_outputs[numTrainingsets][numOutputs] = {{0.0},
                                                            {1.0},
                                                            {1.0},
                                                            {0.0}};

    // Initialize weights and biases
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

    int trainingSetOrder[] = {0, 1, 2, 3};
    int numberOfEpochs = 10000;

    // Training loop
    for (int epoch = 0; epoch < numberOfEpochs; epoch++)
    {
        shuffle(trainingSetOrder, numTrainingsets);

        for (int x = 0; x < numTrainingsets; x++)
        {
            int i = trainingSetOrder[x];

            // Forward pass
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

            // Print output log
            printf("Epoch: %d, Input: [%g, %g], Expected Output: [%g], Predicted Output: [%g]\n",
                   epoch, training_inputs[i][0], training_inputs[i][1],
                   training_outputs[i][0], outputLayer[0]);

            // Backpropagation
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++)
            {
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

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

    return 0;
}
