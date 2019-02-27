import numpy as np
from SimpleNN import FCLayer
from SimpleNN import LossFun

class NeuralNet(object):
    def __init__(self, num_inputnodes,
                 loss_function = 'NLL'):
        self.layers = []
        self.Loss = LossFun.Loss(loss_function)
        self.LossVal = 0
        self.inputlayer = np.zeros(num_inputnodes, dtype=float)
        self.iter = 0

    def addlayer(self, num_node, layertype = 'FC',
                 activation_function = 'relu',
                 optimizer = 'SGD',
                 learning_rate = 0.01, alpha = 0.01,
                 param1 = 0.01, param2 = 0.01):
        if len(self.layers) == 0:
            num_inputnode = len(self.inputlayer)
        else:
            num_inputnode = self.layers[len(self.layers) - 1].num_nodes
        if layertype == 'FC':
            self.layers.append(
                FCLayer.Layer(num_node,
                              num_inputnode,
                              optimizer = optimizer,
                              learning_rate = learning_rate,
                              layertype = layertype,
                              activation_function = activation_function,
                              alpha = alpha,
                              param1=param1,
                              param2=param2))

    def dellayer(self, layerindex):
        self.layers.pop(layerindex)
        if layerindex <= len(self.layers) - 1:
            if len(self.layers) == 0:
                num_inputnode = len(self.inputlayer)
            else:
                num_inputnode = self.layers[layerindex - 1].num_nodes
            self.layers[layerindex].reset(num_inputnode)

    def getloss(self, input, target):
        self.LossVal = self.Loss.compute_loss(
            self.test(input), target)
        return self.LossVal

    def test(self, input):
        temp_input = input
        temp_output = 0
        for layerindex in range(len(self.layers)):
            temp_output = self.layers[layerindex].forwardprop(temp_input)
            temp_input = np.copy(temp_output)
        return temp_output

    def train(self, input, target, minibatch = False):
        self.iter += 1
        if minibatch:
            acc_grad = 0
            self.LossVal = 0
            for s in range (len(input)):
                gradient = self.Loss.compute_grad(
                    self.test(input[s]), target[s])
                acc_grad += gradient
                for layerindex in reversed(range(len(self.layers))):
                    gradient = self.layers[layerindex].backprop(
                        gradient, False, self.iter,
                        minibatch = minibatch)

            gradient = acc_grad / len(input)
            for layerindex in reversed(range(len(self.layers))):
                gradient = self.layers[layerindex].backprop(
                    gradient, True, self.iter,
                    minibatch = minibatch, minibatchsize = len(input))
        else:
            gradient = self.Loss.compute_grad(
                self.test(input), target)
            # print(gradient)
            for layerindex in reversed(range(len(self.layers))):
                gradient = self.layers[layerindex].backprop(
                    gradient, True, self.iter,
                    minibatch = minibatch, minibatchsize = len(input))