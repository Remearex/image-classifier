import numpy as np
import random

from stages import(
    StageTransition,
    MultiClassLogistics
)

from funobjs import(
    Relu
)

class NeuralNetwork:

    def __init__(self, stageSizes):
        self.__stages = []
        self.__lastStage = None

        # initialize all the stages
        for i in range(1, len(stageSizes) - 1):
            nextStageSize = stageSizes[i]
            prevStageSize = stageSizes[i-1]
            stddev = np.sqrt(2/prevStageSize)
            newWeights = np.random.normal(loc=0, scale=stddev, size=(nextStageSize, prevStageSize))
            biasInitConst = 10
            newBias = np.full(nextStageSize, biasInitConst)

            newStage = StageTransition(nextStageSize, prevStageSize, newWeights, newBias, Relu())
            self.__stages.append(newStage)
        
        lastStageInputSize = stageSizes[-2]
        lastStageOutputSize = stageSizes[-1]
        lastStageWeights = np.random.normal(loc=0, scale=np.sqrt(2/lastStageInputSize), size=(lastStageOutputSize, lastStageInputSize))
        lastStageBiasInitConst = 10
        lastStageBias = np.full(lastStageOutputSize, lastStageBiasInitConst)
        self.__lastStage = MultiClassLogistics(lastStageOutputSize, lastStageInputSize, lastStageWeights, lastStageBias)

        # manually set the weights and biases for each stage. This section is used for testing only
        # self.__stages[0].setWeights(np.array([[0.82026296, -1.34552759], [0.28820175, -1.83752342], [0.42797499, -1.36308585]]))
        # self.__stages[0].setBias(np.array([10.00007206, 9.91837515, 9.89200139]))
        # self.__stages[0].setBias(np.zeros(3))
        # self.__stages[1].setWeights(np.array([[-1.32059796, 0.40481362, -1.11406219], [-0.8585316, -1.12633172, -0.23812984], [-0.1424315, -0.12387872, -1.83013595]]))
        # self.__stages[1].setBias(np.array([9.92059801, 9.84446532, 10]))
        # self.__stages[1].setBias(np.zeros(3))
        # self.__lastStage.setWeights(np.array([[0.689080328, -0.000802711112, -1.52996461], [0.286563864, 0.571241596, -1.40057758]]))
        # self.__lastStage.setBias(np.array([10, 10]))
        # self.__lastStage.setBias(np.zeros(2))


    def fit(self, Xtrain, ytrain, maxGDIter, sgdBatchSize, sgdStepSize, verbose):
        for i in range(maxGDIter):
            indices = list(range(len(Xtrain)))
            random.shuffle(indices)
            selected_indices = indices[:sgdBatchSize]
            selected_Xtrain = [Xtrain[j] for j in selected_indices]
            selected_ytrain = [ytrain[j] for j in selected_indices]

            # params
            # print("params:")
            # print(self.__stages[0].getWeights())
            # print(self.__stages[0].getBias())
            # print(self.__stages[1].getWeights())
            # print(self.__stages[1].getBias())
            # print(self.__lastStage.getWeights())
            # print(self.__lastStage.getBias())

            #calculate gradient for each selected example and update weights
            tot_loss = 0
            for j in range(sgdBatchSize):
                example = selected_Xtrain[j]
                label = selected_ytrain[j]
                #forward prop
                temp = example
                # print("temp =", temp)
                for k in range(len(self.__stages)):
                    temp = self.__stages[k].forward(temp)
                    # print("temp =", temp)
                tot_loss += self.__lastStage.forward(temp, label)
                #backward prop
                temp = self.__lastStage.backward(label)
                # print("temp =", temp)
                for k in range(len(self.__stages)-1, -1, -1):
                    temp = self.__stages[k].backward(temp)
                    # print("temp =", temp)

            if verbose:
                print("loss over", sgdBatchSize, "examples =", tot_loss)
            
            # gd step
            # print("gd stepping:")
            for k in range(len(self.__stages)):
                self.__stages[k].gdStep(sgdStepSize/sgdBatchSize)
            self.__lastStage.gdStep(sgdStepSize/sgdBatchSize)

    def predict(self, Xtrain):
        temp = Xtrain.T
        for k in range(len(self.__stages)):
            temp = self.__stages[k].forwardAll(temp)
        result = self.__lastStage.forwardAll(temp)
        print(result)