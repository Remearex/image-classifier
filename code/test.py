import numpy as np

from neural_network import (
    NeuralNetwork
)

from stages import (
    StageTransition,
    MultiClassLogistics,
)

from funobjs import(
    Relu
)

from keras.datasets import mnist

class MockFunObj:
    def eval(self, next_z):
        return next_z
    
    def deriv(self, next_z):
        return next_z
        

def backward_test():
    mock_fun_obj = MockFunObj()
    st = StageTransition(3, 4, mock_fun_obj)
    st.setWeights(np.array([[7, 9, 5, 4], [3, 1, 8, 6], [5, 2, 9, 3]]))
    st.setBias(np.array([9, 5, 7]))
    st.forward(np.array([12, 5, 7, 3]))
    prevdfda = st.backward(np.array([3, 7, 5]))
    weights_partials = st.getWeightsPartials()
    bias_partials = st.getBiasPartials()
    print(prevdfda)
    print(weights_partials)
    print(bias_partials)

def hiddenStageLastStage_test():
    relu = Relu()
    x_i = np.array([1, 2])
    y_i = 1
    W1 = np.array([[3, 2], [1, 3], [2, 4]])
    b1 = np.array([1, -1, 1])
    W2 = np.array([[1, 2, 1], [2, 1, 3], [2, 2, 1]])
    b2 = np.array([-25, -50, -32])
    hiddenStage = StageTransition(3, 2, W1, b1, relu)
    lastStage = MultiClassLogistics(3, 3, W2, b2)

    # forward prop x_i
    next_a = hiddenStage.forward(x_i)
    loss = lastStage.forward(next_a, y_i)

    # backward prop
    dfda1 = lastStage.backward(y_i)
    dfda2 = hiddenStage.backward(dfda1)

    # check vals
    print(loss)
    print(lastStage.getWeightsPartials())
    print(lastStage.getBiasPartials())
    print(dfda1)
    print(hiddenStage.getWeightsPartials())
    print(hiddenStage.getBiasPartials())
    print(dfda2)


# pretends that the data matrix has 2 copies of the same example and same labels
# this is just to check that the gradients add up properly when we push multiple
# examples through the neural net
def hiddenStageLastStageMultiExample_test():
    relu = Relu()
    x_i = np.array([1, 2])
    y_i = 1
    W1 = np.array([[3, 2], [1, 3], [2, 4]])
    b1 = np.array([1, -1, 1])
    W2 = np.array([[1, 2, 1], [2, 1, 3], [2, 2, 1]])
    b2 = np.array([-25, -50, -32])
    hiddenStage = StageTransition(3, 2, W1, b1, relu)
    lastStage = MultiClassLogistics(3, 3, W2, b2)

    # forward prop on first example
    next_a = hiddenStage.forward(x_i)
    loss = lastStage.forward(next_a, y_i)

    # backward prop on first example
    dfda1 = lastStage.backward(y_i)
    dfda2 = hiddenStage.backward(dfda1)

    # forward prop on second example (still x_i)
    next_a = hiddenStage.forward(x_i)
    loss = lastStage.forward(next_a, y_i)

    # backward prop on second example (still x_i)
    dfda1 = lastStage.backward(y_i)
    dfda2 = hiddenStage.backward(dfda1)

    # check vals
    print(loss)
    print(lastStage.getWeightsPartials())
    print(lastStage.getBiasPartials())
    print(hiddenStage.getWeightsPartials())
    print(hiddenStage.getBiasPartials())
    print(dfda2)

def neuralNet_test():
    neuralNetwork = NeuralNetwork([2, 3, 3])
    X = np.array([[1, 2], [1, 2]])
    y = np.array([1, 1])
    neuralNetwork.fit(X, y, 1, 2, 2, True)

def neuralNetXOR_test():
    neuralNetwork = NeuralNetwork([2, 4, 2])
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    neuralNetwork.fit(X, y, 1000, 3, 0.01, True)
    neuralNetwork.predict(X)

def mnist_test():
    (Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()

    print('X_train: ' + str(Xtrain.shape))
    print('Y_train: ' + str(ytrain.shape))
    print('X_test:  '  + str(Xtest.shape))
    print('Y_test:  '  + str(ytest.shape))

    Xtrain = Xtrain.reshape(len(Xtrain), 784)
    Xtest = Xtest.reshape(len(Xtest), 784)

    neuralNetwork = NeuralNetwork([784, 784, 784, 10])
    neuralNetwork.fit(Xtrain, ytrain, 1000, 100, 10, True)


def test():
    # hiddenStageLastStageMultiExample_test()
    # neuralNet_test()
    # neuralNetXOR_test()
    mnist_test()

    