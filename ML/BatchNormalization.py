import numpy as np
from Layer import Layer


class BatchNormalization(Layer):
    def __init__(self, inputSize, outputsize, updateStrategy):
        self.weight = np.zeros((2, 2))
        self.batchVar = None
        self.batchMean = None
        self.shiftedData = None
        self.inputData = None
        self.inputSize = inputSize
        self.gama = np.ones(inputSize)
        self.beta = np.zeros(inputSize)
        if inputSize != outputsize:
            raise ValueError("InputSize and outputSize don't match")
        #### Accumulate mean and variance######
        self.mean = 0
        self.variance = 0
        self.m = 0
        #######################################

    def forward(self, upstreamResult):
        batchsize = upstreamResult.shape[0]
        self.inputData = upstreamResult
        self.batchMean = np.mean(upstreamResult, axis=0)
        self.batchVar = np.mean(np.power(upstreamResult-self.batchMean, 2), axis=0)
        self.shiftedData = (upstreamResult-self.batchMean)/np.sqrt(self.batchVar + 1e-8)
        self.variance = (self.variance * self.m + self.batchVar * batchsize) / (self.m + batchsize)
        self.mean = (self.mean * self.m + self.batchMean * batchsize) / (self.m + batchsize)
        self.m += batchsize
        return self.gama*self.shiftedData + self.beta

    def backward(self, downstreamG, learningRate, lambd):
        batchSize = downstreamG.shape[0]
        gSD = downstreamG * self.gama
        gVar = np.sum(gSD*(self.batchMean - self.inputData)/(2*np.power(self.batchVar + 1e-8, 1.5)), axis=0)
        gMean = np.sum((-1*gSD/np.sqrt(self.batchVar + 1e-08)) + 2*gVar*(self.batchMean - self.inputData)/batchSize, axis=0)
        self.gama -= learningRate * np.sum(downstreamG * self.shiftedData, axis=0)
        self.beta -= learningRate * np.sum(downstreamG, axis=0)
        return gSD / np.sqrt(self.batchVar + 1e-8) + (2 * gVar * (self.inputData - self.batchMean) + gMean) / batchSize

    def predict(self, upstreamResult):
        return self.gama * (upstreamResult - self.mean) / np.sqrt(self.variance + 1e-8) + self.beta