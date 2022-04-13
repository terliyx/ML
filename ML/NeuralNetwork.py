import numpy as np
from Agent import Agent
from PlainInputLayer import PlainInputLayer
from SigmoidLayer import SigmoidLayer
from SoftmaxLayer import SoftmaxLayer
from ReLULayer import ReLULayer
from BatchNormalization import BatchNormalization

availableLayers = {"P": PlainInputLayer, "Sig": SigmoidLayer, "Sof": SoftmaxLayer, "ReLU": ReLULayer, "BN": BatchNormalization}
# currently available layers are:
#     input layer: P - PlainInputLayer
#     hidden layer: Sig - SigmoidLayer
#                   ReLU - ReLULayer
#     output layer: Sof - SoftmaxLayer
#     auxiliary layer:BN - BatchNormalization


class NeuralNetwork(Agent):
    def __init__(self, agentName, dataDemension, bias=0):
        self.layers = []
        self.bias = bias
        super().__init__(agentName)
        self.size = 0
        self.dataDemension = dataDemension

    def train(self, data, y, classes, iteration, lambd, learningRate):
        processedData = self.processData(data, self.bias)
        m_y = self.processY(y, classes)
        for i in range(iteration):
            self.feedforward(processedData)
            self.backprop(m_y, learningRate, lambd)
            print("After " + str(i) + "th iteration, loss is " + str(self.getLoss(m_y, lambd)) + "(" + str(self.getLoss(m_y, 0)))
        #self.saveAgent()

    def predict(self, data):
        biasedData = self.processData(data, self.bias)
        result = biasedData
        for layer in self.layers:
            result = layer.predict(result)
        return np.argmax(result, axis=1)

    def loadAgent(self):
        f = open(self.agentName, "br+")
        self.layers = list(np.load(f, allow_pickle=True))
        f.close()
        print("layers loaded")
        self.size = len(self.layers)
        print("this is a " + str(self.size-2) + "-layer NN")

    def getLoss(self, m_y, lambd):
        return self.layers[-1].getLoss(m_y, self.layers, lambd)

    def gradientCheck(self, dat, y, classes, delta):
        data = self.processData(dat, self.bias)
        self.feedforward(data)
        m_y = self.processY(y, classes)
        highestE = 0
        AGs = []
        gradient = self.layers[-1].backward(m_y)
        for layer in self.layers[-2::-1]:
            gradients = layer.backwardForGC(gradient)
            gradient = gradients[1]
            AGs.insert(0, gradients[0])
        n = 0
        for layer in self.layers[0:-1]:
            AG = AGs[n]
            print(str(n) + "th layer")
            for i in range(layer.weight.shape[0]):
                for j in range(layer.weight.shape[1]):
                    layer.weight[i][j] += delta
                    self.feedforward(data)
                    pLoss = self.getLoss(m_y, 0)
                    layer.weight[i][j] -= 2 * delta
                    self.feedforward(data)
                    nLoss = self.getLoss(m_y, 0)
                    layer.weight[i][j] += delta
                    NG = (pLoss-nLoss)/(2*delta)
                    print((NG, AG[i][j]))
                    if NG != 0 or AG[i][j] != 0:
                        E = abs(NG-AG[i][j])/max(abs(NG), abs(AG[i][j]))
                        if E > highestE:
                            highestE = E
                            print("max E " + str(E) + " at (" + str(i) + "," + str(j) + ")")
            n += 1



    def getLossRawY(self, y, classes,  lambd):
        m_y = self.processY(y, classes)
        return self.layers[-1].getLoss(m_y, self.layers, lambd)

    def setUpLayers(self, layerConfig):
        # layerConfig is a list contains layer configs that decide what layers to be added in this NN,
        # layer config takes a form of (layer name, # nodes), except output layer which is (layer name),
        # input layer shall be the first and output layer shall be the last.
        curInputSize = self.dataDemension
        for config in layerConfig[0:-1]:
            layerConstructor = availableLayers.get(config[0])
            self.layers.append(layerConstructor(curInputSize, config[1]))
            curInputSize = config[1]
            self.size += 1
        layerConstructor = availableLayers.get(layerConfig[-1])
        self.layers.append(layerConstructor())
        self.size += 1

    def backprop(self, y, learningRate, lambd):
        # y have a shape N*k
        gradient = self.layers[-1].backward(y)
        for layer in self.layers[-2::-1]:
            gradient = layer.backward(gradient, learningRate, lambd)

    def backprops(self, downstream, learningRate, lambd, curL=-1):
        # recursive implementation (under-tested)
        if curL == -1 * self.size:
            return self.layers[curL].backward(downstream, learningRate, lambd)
        elif curL == -1:
            return self.backprops(self.layers[curL].backward(downstream), learningRate, lambd, curL - 1)
        else:
            return self.backprops(self.layers[curL].backward(downstream, learningRate, lambd), learningRate, lambd, curL-1)

    def feedforward(self, data):
        result = data
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def feedforwards(self, upstream, curL=0):
        # recursive implementation (under-tested)
        if curL == self.size-1:
            return self.layers[curL].forward(upstream)
        else:
            return self.feedforwards(self.layers[curL].forward(upstream), curL+1)

    def saveAgent(self):
        for layer in self.layers:
            layer.passedData = None
        f = open(self.agentName, "bw+")
        np.save(f, np.array(self.layers))
        f.close()
        print("layers saved")

