from Agent import Agent
import numpy as np


def softmaxL(predic, y, weight, lambd):
    # y: ground truth,size of N*k; predic: prediction from X*Weight, size of N*k;

    loss = np.sum(-1 * np.sum(y * predic, axis=1) + np.sum(np.log(np.sum(np.exp(predic), axis=1)))) / (predic.shape[
        0]**2) + 0.5 * lambd * np.sum(np.square(weight[1:, :]))
    return loss


def softmaxG(data, predic, y, weight, lambd):
    # gradient of softmax loss,
    (N, classes) = predic.shape
    exp_p = np.exp(predic)
    dt = data.T
    denominator = (np.sum(exp_p, axis=1, keepdims=True))
    lossG = (dt.dot(exp_p / denominator - y))/(N**2) + lambd * np.append(np.zeros((1, weight.shape[1])), weight[1:, :], axis=0)
    return lossG


class LinearClassifier(Agent):
    def __init__(self, agentName):
        self.weight = np.array([])
        super().__init__(agentName)

    def train(self, data, y, classes, iteration, lambd=0.01, lossF="softmax", learningRate=0.001):
        # data should be a N*m matrix with N examples in which each example have m features and is hold in one row.
        # lambd: regularization coefficient, default value is not tested best, also for default learning rate.
        if lossF == "softmax":
            loss = softmaxL
            gradient = softmaxG
        biasedData = self.processData(data)
        m_y = self.processY(y, classes)
        if self.weight.size == 0:
            # weight matrix: size of m+1*k, where m is the number of features for an unprocessed example,
            # and k is the number of classes.
            self.weight = (np.random.random((biasedData.shape[1], m_y.shape[1])) * 2) - 1
        for i in range(iteration):
            predic = biasedData.dot(self.weight)
            contained_predic = predic - np.max(predic, axis=1, keepdims=True)
            G = gradient(biasedData, contained_predic, m_y, self.weight, lambd/learningRate)

            self.weight -= learningRate * G
            #print("after " + str(i+1) + "th iteration, loss  " + str(loss(contained_predic, m_y, self.weight, lambd)))
        self.saveAgent()

    def predict(self, data):
        biasedData = self.processData(data)
        predic = biasedData.dot(self.weight)
        return np.argmax(predic, axis=1)

    def loadAgent(self):
        f = open(self.agentName, "br+")
        self.weight = np.load(f)
        print("weight loaded")
        f.close()

    def saveAgent(self):
        f = open(self.agentName, "bw+")
        np.save(f, self.weight)
        print("weight saved")
        f.close()

    def getLoss(self, data, y, classes, lambd, lossF="softmax"):
        if lossF == "softmax":
            loss = softmaxL
        biasedData = self.processData(data)
        m_y = self.processY(y, classes)
        predic = biasedData.dot(self.weight)
        contained_predic = predic - np.max(predic, axis=1, keepdims=True)
        return loss(contained_predic, m_y, self.weight, lambd)
