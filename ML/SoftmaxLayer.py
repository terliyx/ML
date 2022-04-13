import numpy as np

from Layer import Layer

def softmax(scores):
    # softmax activation function
    return np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)


def softmaxG(predic, y):
    # gradient of softmax loss
    return (predic - y) / y.shape[0]


class SoftmaxLayer(Layer):
    # output layer, don't have a weight matrix, forward() will softmax input and return a final prediction directly.
    # backward() will use softmax loss function to calculate gradient for the previous layer.
    def __init__(self):
        self.passedData = None

    def forward(self, upstreamResult):
        # save a contained prediction
        containedR = upstreamResult - np.max(upstreamResult, axis=1, keepdims=True)
        self.passedData = softmax(containedR)
        return self.passedData

    def predict(self, upstreamResult):
        containedR = upstreamResult - np.max(upstreamResult, axis=1, keepdims=True)
        self.passedData = softmax(containedR)
        return self.passedData

    def backward(self, y):
        return softmaxG(self.passedData, y)


    def getLoss(self, y, layers, lambd):
        # y: ground truth,size of N*k
        # add 0.000000000001 for numerical stability(under-tested)
        loss = np.sum(-1 * np.log(np.sum(self.passedData * y, axis=1) + 0.0000000001)) / self.passedData.shape[0]
        for layer in layers[0:-1]:
            loss += 0.5 * lambd * np.sum(np.square(layer.weight[1:, :]))
        return loss
