A machine learning model that classifies images and other data

Uses a fully connected neural network where each stage transition is represented by a class

For forward propagation, each stage transition takes as input the output of the previous layer, performs the matrix multiplication, non-linear activation function, and returns the output.

For back propagation, each stage transition takes as input the partial derivatives of the loss function with respect to each of the outputs of the stage transition, then calculates the partials wrt the weights and biases, and outputs the partials of the loss function wrt the inputs of this stage transition.