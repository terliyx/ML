import numpy as np


class Agent(object):
    # N: total number of samples; m: length of one sample; k: number of classes;
    def __init__(self, agentName):
        # check if it should create a file for saving the agent
        self.agentName = agentName
        try:
            self.f = open(agentName, "br+")
            print("found existing agent, use loadAgent() to load in or create a new one without loading")

        except FileNotFoundError:
            self.f = open(agentName, "bw+")
            print("no existing agent, a new one has been created")
        self.f.close()

    def processY(self, y, classes):
        m_y = np.zeros((y.shape[0], classes))
        for i in range(y.size):
            m_y[i][y[i]] = 1
        return m_y

    def processData(self, data, bias=0):
        # rescale and add biased terms to data.
        biasedData = self.addBias(self.rescaleData(data), bias)
        return biasedData

    def rescaleData(self, data):
        dataMax = np.max(data)
        return data / dataMax - 0.5

    def addBias(self, data, bias=0):
        # add biased terms (ie. 1s) to every examples in data
        biasedData = np.hstack((bias*np.ones((data.shape[0], 1)), data))
        return biasedData
