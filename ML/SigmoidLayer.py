import numpy as np

from Layer import Layer


def sigmoid(upstreamResult):
    return np.reciprocal(1 + np.exp(-1 * upstreamResult))


class SigmoidLayer(Layer):
    def __init__(self, inputSize, outputSize, updateStrategy):
        self.weight = np.random.randn(inputSize+1, outputSize)/np.sqrt(inputSize+1)
        self.update = Layer.updateStrategies(updateStrategy)
        self.passedData = None

    def forward(self, upstreamResult):
        # receive data or result forwarded down by the previous layer, add bias terms, sigmoid and save it,
        # return a result ready to be forward down.
        self.passedData = self.addBias(sigmoid(upstreamResult))
        return self.passedData.dot(self.weight)

    def predict(self, upstreamResult):
        return self.addBias(sigmoid(upstreamResult)).dot(self.weight)

    def backward(self, downstreamG, learningRate, lambd):
        # receive a gradient with a size same as passedData.dot(weight) from it's next layer,
        # use it to update weight and backprop a gradient to it's previous layer (without newly added bias terms).
        self.weight -= (learningRate * self.passedData.T.dot(downstreamG) + lambd * np.vstack((np.zeros((1, self.weight.shape[1])), self.weight[1:, :])))
        return ((self.passedData * (1 - self.passedData)) * downstreamG.dot(self.weight.T))[:, 1:]


    def backwardForGC(self, downstreamG):
        return (self.passedData.T.dot(downstreamG), ((self.passedData * (1 - self.passedData)) * downstreamG.dot(self.weight.T))[:, 1:])

    def addBias(self, data):
        # add bias terms (ie. 0s) to every examples in data
        biasedData = np.hstack((np.zeros((data.shape[0], 1)), data))
        return biasedData