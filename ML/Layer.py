import numpy as np


def SGD(layer, downstreamG):
    return layer.passedData.T.dot(downstreamG)


def SGDMomentum(layer, downstreamG):
    layer.velocity = 0.9 * layer.velocity + layer.passedData.T.dot(downstreamG)
    return layer.velocity


def AdaGrad(layer, downstreamG):
    layer.squaredG += np.square(layer.passedData.T.dot(downstreamG))
    return layer.passedData.T.dot(downstreamG)/(np.sqrt(layer.squaredG) + 1e-08)


def RMSProp(layer, downstreamG):
    layer.squaredG = 0.9 * layer.squaredG + 0.1 *np.square(layer.passedData.T.dot(downstreamG))
    return layer.passedData.T.dot(downstreamG) / (np.sqrt(layer.squaredG) + 1e-08)


def Adam(layer, downstreamG):
    layer.firstM = 0.9 * layer.firstM + 0.1 * layer.passedData.T.dot(downstreamG)
    layer.secondM = 0.999 * layer.secondM + 0.001 * np.square(layer.passedData.T.dot(downstreamG))
    return layer.firstM / (layer.secondM + 1e-08)


updateStrategies = {"SGD": SGD, "SGDM": SGDMomentum, "AdaG": AdaGrad, "RMSP": RMSProp, "Adam": Adam}


class Layer(object):

    def addBias(self, data, bias=0):
        # add bias terms to every examples in data
        biasedData = np.hstack((bias*np.ones((data.shape[0], 1)), data))
        return biasedData


