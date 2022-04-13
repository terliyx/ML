import numpy as np
from Layer import Layer


class ReLULayer(Layer):
    def __init__(self, inputSize, outputSize, updateStrategy):
        self.weight = np.random.randn(inputSize + 1, outputSize)/np.sqrt((inputSize+1)/2)
        self.update = Layer.updateStrategies(updateStrategy)
        ###################################
        # comment out this line
        self.passedData = None
        ###################################
        # then uncomment these lines, do the same thing in backward() to reserve the data in self.passedData
        #self.passedData = None
        #self.activated = None
        ###################################

    def forward(self, upstreamResult):
        # receive data or result forwarded down by the previous layer, add bias terms, rectify and save it,
        # return a result ready to be forward down.
        self.passedData = self.addBias(np.maximum(0, upstreamResult))
        return self.passedData.dot(self.weight)

    def predict(self, upstreamResult):
        return self.addBias(np.maximum(0, upstreamResult)).dot(self.weight)

    def backward(self, downstreamG, learningRate, lambd):
        # receive a gradient with a size same as passedData.dot(weight) from it's next layer,
        # use it to update weight and backprop a gradient to it's previous layer (without newly added bias terms).
        self.activated = np.zeros(self.passedData.shape)
        self.activated[self.passedData > 0] = 1
        g = (self.activated * downstreamG.dot(self.weight.T))[:, 1:]

        self.weight -= (learningRate * self.passedData.T.dot(downstreamG) + lambd * np.vstack((np.zeros((1, self.weight.shape[1])), self.weight[1:, :])))
        return g
        #########################################
        #self.passedData[self.passedData > 0] = 1
        #return (self.passedData * downstreamG.dot(self.weight.T))[:, 1:]
        ########################################
        #self.activated = np.zeros(self.passedData.shape)
        #self.activated[self.passedData > 0] = 1
        #return (self.activated * downstreamG.dot(self.weight.T))[:, 1:]
        #########################################

    def backwardForGC(self, downstreamG):
        self.activated = np.zeros(self.passedData.shape)
        self.activated[self.passedData > 0] = 1
        g = (self.activated * downstreamG.dot(self.weight.T))[:, 1:]
        return (self.passedData.T.dot(downstreamG), g)

