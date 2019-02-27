import numpy as np
from SimpleNN import Optimizer
from SimpleNN import Activate

class Layer():

    def __init__(self, num_nodes, num_inputnodes,
                 optimizer, learning_rate,
                 layertype, activation_function,
                 alpha, param1, param2):
        self.num_nodes = num_nodes
        self.num_inputnodes = num_inputnodes
        self.type = layertype
        self.Act = Activate.Activate(activation_function, alpha)
        self.activation_function = activation_function
        self.Optimizer = Optimizer.Optimizer(optimizer, learning_rate,
                                             param1, param2)
        self.temp_in = 0
        self.temp_act = 0
        self.grad_w = 0
        self.W_xh = np.random.random(
            (self.num_nodes, self.num_inputnodes)) * (2 / np.sqrt(num_inputnodes)) - (1 / np.sqrt(num_inputnodes))
        self.b_h = np.random.random((num_nodes)) * (2 / np.sqrt(num_inputnodes)) - (1 / np.sqrt(num_inputnodes))

    def reset(self, num_inputnodes):
        self.num_inputnodes = num_inputnodes
        self.W_xh = np.random.rand(self.num_nodes, self.num_inputnodes)

    def setparam(self, W_xh, b_h):
        self.W_xh = W_xh
        self.b_h = b_h

    def getparam(self):
        return self.W_xh, self.b_h

    def forwardprop(self, input):
        if len(input[0]) != self.W_xh.shape[1]:
            print("input size doesn't match!")
            return 0
        # if there is no maxout, then just use np.dot
        if self.activation_function != 'maxout':
            node_temp = np.dot(input, np.transpose(self.W_xh))
            node_temp = np.add(node_temp, self.b_h)
            node_out = self.Act.activate(node_temp, 'forward')
        else:
            node_temp = [np.multiply(input, self.W_xh[i])
                         for i in range(self.num_nodes)]
            node_out = self.Act.activate(node_temp, 'forward')
            node_out = np.add(node_out, self.b_h)
        self.temp_in = input
        self.temp_act = node_temp
        return node_out

    def backprop(self, gradient, train, iter, minibatch, minibatchsize = 1):
        act_gradient = self.Act.activate(self.temp_act, 'backward')
        local_gradient = np.multiply(act_gradient, gradient)
        prop_gradient = np.dot(local_gradient, self.W_xh)

        if minibatch:
            self.grad_w += np.dot(np.transpose(self.temp_in), local_gradient).T
        else:
            self.grad_w = np.dot(np.transpose(self.temp_in), local_gradient).T

        if train:
            self.grad_w = self.grad_w / minibatchsize
            self.W_xh, self.b_h = \
                self.Optimizer.update_layer(self.W_xh,
                                            self.b_h,
                                            self.grad_w,
                                            local_gradient,
                                            iter)
            self.grad_w = 0
        return prop_gradient