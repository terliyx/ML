import numpy as np

from Layer import Layer


class PlainInputLayer(Layer):
    # a plain input layer.
    def __init__(self, dataDemension, outputSize, updateStrategy):
        self.weight = np.random.randn(dataDemension+1, outputSize)/np.sqrt(dataDemension+1)
        self.update = Layer.updateStrategies(updateStrategy)
        self.passedData = None

    def forward(self, data):
        # incoming data should be preprocessed.
        # save data for later backprop.
        self.passedData = data
        return data.dot(self.weight)

    def predict(self, data):
        return data.dot(self.weight)

    def backward(self, downstreamG, learningRate, lambd):
        # update weight matrix, return nothing.

        self.weight -= (learningRate * self.passedData.T.dot(downstreamG) + lambd * np.vstack((np.zeros((1, self.weight.shape[1])), self.weight[1:, :])))


    def backwardForGC(self, downstreamG):
        return (self.passedData.T.dot(downstreamG), 0)
