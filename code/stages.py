import numpy as np

class StageTransition:
    
    def __init__(self, nextStageSize, prevStageSize, weights, bias, funobj):
        self.__weights = weights.astype(float)
        self.__bias = bias.astype(float)
        heightw, widthw = weights.shape
        assert heightw == nextStageSize and widthw == prevStageSize and (bias.ndim == 1 or bias.ndim == 2)
        # if bias is a col vector, convert it to a row vector
        if bias.ndim == 2:
            assert bias.shape[1] == 1
            bias = bias.flatten()
        assert len(bias) == nextStageSize
        self.__weights_partials = np.zeros((nextStageSize, prevStageSize))
        self.__bias_partials = np.zeros(nextStageSize)
        self.__prev_a = np.zeros(prevStageSize)
        self.__next_z = np.zeros(nextStageSize)
        self.funobj = funobj
        self.__prevStageSize = prevStageSize
        self.__nextStageSize = nextStageSize
    
    def backward(self, nextdfda):  
        z_derivatives = self.funobj.deriv(self.__next_z)
        azprime_product = nextdfda * z_derivatives

        self.__bias_partials += azprime_product
        
        self.__weights_partials += np.outer(azprime_product, self.__prev_a)

        temp = self.__weights * azprime_product[:, np.newaxis]
        return temp.sum(axis=0)
    
    def forward(self, prev_a):
        self.__prev_a = prev_a
        # print("prev_a =", prev_a)

        next_z = self.__weights @ prev_a + self.__bias
        # print("next_z =", next_z)
        self.__next_z = next_z

        temp = self.funobj.eval(next_z)
        # print("next_a =", temp)
        return temp
    
    def forwardAll(self, prev_A):
        # add self.__bias as a new col in self.__weights, and add a new row of 1s to prev_A then multiply and apply z_to_a_funobj
        new_weights = np.hstack((self.__weights, self.__bias[:, np.newaxis]))
        row_of_ones = np.ones(prev_A.shape[1], dtype=prev_A.dtype)
        new_prev_A = np.vstack((prev_A, row_of_ones))
        return self.funobj.eval(new_weights @ new_prev_A)

    
    def gdStep(self, alpha):
        # print("weights partisl =")
        # print(self.__weights_partials)
        # print("bias partials =")
        # print(self.__bias_partials)
        self.__weights -= alpha * self.__weights_partials
        self.__bias -= alpha * self.__bias_partials
        self.__weights_partials = np.zeros((self.__nextStageSize, self.__prevStageSize))
        self.__bias_partials = np.zeros(self.__nextStageSize)
        # print("new weights =", self.__weights)
        # print("new bias =", self.__bias)
    
    def setWeights(self, weights):
        self.__weights = weights.astype(float)

    def setBias(self, bias):
        self.__bias = bias.astype(float)

    def getWeightsPartials(self):
        return self.__weights_partials
    
    def getBiasPartials(self):
        return self.__bias_partials
    
    def getWeights(self):
        return self.__weights
    
    def getBias(self):
        return self.__bias
    




class LastStage:

    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
    
    def backward(self):
        pass

    def forward(self, prev_a):
        pass

    def forwardAll(self, prev_A):
        pass



class MultiClassLogistics(LastStage):

    def __init__(self, outputSize, inputSize, weights, bias):
        super().__init__(inputSize, outputSize)
        self.__weights = weights.astype(float)
        self.__bias = bias.astype(float)
        heightw, widthw = weights.shape
        assert heightw == outputSize and widthw == inputSize and (bias.ndim == 1 or bias.ndim == 2)
        # if bias is a col vector, convert it to a row vector
        if bias.ndim == 2:
            assert bias.shape[1] == 1
            bias = bias.flatten()
        assert len(bias) == outputSize
        self.__weights_partials = np.zeros((outputSize, inputSize))
        self.__bias_partials = np.zeros(outputSize)
        self.__o = np.zeros(outputSize)
        self.__prev_a = np.zeros(inputSize)
        self.__outputSize = outputSize
        self.__inputSize = inputSize

    def backward(self, y_i):
        dfdo = np.exp(self.__o) / sum(np.exp(self.__o))
        dfdo[y_i] -= 1
        # print("dfdo =", dfdo)

        self.__bias_partials += dfdo

        product = np.outer(dfdo, self.__prev_a)
        self.__weights_partials += np.outer(dfdo, self.__prev_a)

        temp = self.__weights * dfdo[:, np.newaxis]
        return temp.sum(axis=0)
    
    # caches self.__prev_a and self.__o and returns the loss for this example
    def forward(self, prev_a, y_i):
        self.__prev_a = prev_a
        # print("last stage prev_a =", prev_a)
        self.__o = self.__weights @ prev_a + self.__bias
        # print("last stage o =", self.__o)
        return -self.__o[y_i] + np.log(sum(np.exp(self.__o)))

    # classifies each example. Returns a vector where the ith entry is the classified
    # digit of the ith example
    def forwardAll(self, prev_A):
        new_weights = np.hstack((self.__weights, self.__bias[:, np.newaxis]))
        row_of_ones = np.ones(prev_A.shape[1], dtype=prev_A.dtype)
        new_prev_A = np.vstack((prev_A, row_of_ones))
        result = new_weights @ new_prev_A
        # print("printing:")
        # print(result)
        return np.argmax(result, axis=0)
    
    def gdStep(self, alpha):
        # print("weights partisl =")
        # print(self.__weights_partials)
        # print("bias partials =")
        # print(self.__bias_partials)
        self.__weights -= alpha * self.__weights_partials
        self.__bias -= alpha * self.__bias_partials
        self.__weights_partials = np.zeros((self.__outputSize, self.__inputSize))
        self.__bias_partials = np.zeros(self.__outputSize)
        # print("new weights =", self.__weights)
        # print("new bias =", self.__bias)

    def setWeights(self, weights):
        self.__weigts = weights.astype(float)

    def setBias(self, bias):
        self.__bias = bias.astype(float)

    def getWeightsPartials(self):
        return self.__weights_partials
    
    def getBiasPartials(self):
        return self.__bias_partials
    
    def getWeights(self):
        return self.__weights
    
    def getBias(self):
        return self.__bias
