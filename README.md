# XOR Neural Network in C

This project implements a simple neural network to learn the XOR function using the C programming language. The neural network is a fully connected feedforward network with one hidden layer.

original tutorial: https://www.youtube.com/watch?v=LA4I3cWkp1E&ab_channel=NicolaiNielsen

## Features

- Implements a neural network with:
  - 2 input nodes
  - 2 hidden nodes
  - 1 output node
- Uses sigmoid activation function
- Trains the network using backpropagation
- Prints the output during training and the final outputs after training

## Getting Started

### Prerequisites

To compile and run the program, you need a C compiler such as GCC.

### Compilation and Execution

1. **Clone the repository or download the source code**.
   
2. **Navigate to the directory** containing the source code.

3. **Compile the program**:
   ```sh
   gcc -o xor_nn xor_nn.c -lm