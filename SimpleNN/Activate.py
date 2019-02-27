import numpy as np

class Activate:
    def __init__(self, activation_funtion, alpha):
        self.activation_function = activation_funtion
        self.alpha = alpha

    def activate(self, node_temp, direction):
        node_out = 0
        if direction != 'forward' and direction != 'backward':
            print("direction on activation is wrong")
            return 0
        if self.activation_function == 'relu':
            if direction == 'forward':
                node_out = np.clip(node_temp, a_min=0, a_max=np.inf)
            elif direction == 'backward':
                node_out = np.where(node_temp > 0, 1, 0)
        elif self.activation_function == 'lrelu':
            if direction == 'forward':
                node_out = np.where(node_temp > 0,
                                    node_temp,
                                    np.multiply(node_temp,self.alpha))
            elif direction == 'backward':
                node_out = np.where(node_temp > 0, 1, self.alpha)
        elif self.activation_function == 'elu':
            s_out = np.multiply(self.alpha,
                                np.subtract(np.exp(node_temp),1))
            if direction == 'forward':
                node_out = np.where(node_temp > 0,
                                    node_temp, s_out)
            elif direction == 'backward':
                node_out = np.where(node_temp > 0, 1,
                                    np.add(s_out,self.alpha))
        elif self.activation_function == 'sigmoid':
            sigmoid = np.divide(1, np.add(1, np.exp(node_temp)))
            if direction == 'forward':
                node_out = sigmoid
            elif direction == 'backward':
                node_out = np.multiply(sigmoid,
                                       np.subtract(1, sigmoid))
        elif self.activation_function == 'tanh':
            tanh = np.tanh(node_temp)
            if direction == 'forward':
                node_out = tanh
            elif direction == 'backward':
                node_out = np.subtract(1,np.power(tanh, 2))
        elif self.activation_function == 'linear':
            if direction == 'forward':
                node_out = node_temp
            elif direction == 'backward':
                node_out = np.ones_like(node_temp)
        elif self.activation_function == 'maxout':
            node_out = np.amax(np.transpose(node_temp), axis=0)
            if direction == 'forward':
                node_out = node_out
            elif direction == 'backward':
                node_out = np.where(node_out == np.amax(node_out,
                                                         axis=0),
                                    1, 0)
        elif self.activation_function == 'softmax':
            e_out = np.exp(node_temp) - np.max(node_temp)
            softmax = e_out / np.sum(e_out)
            if direction == 'forward':
                node_out = softmax
            elif direction == 'backward':
                node_out = np.multiply(softmax,
                                       np.subtract(1, softmax))
        elif self.activation_function == 'softplus':
            if direction == 'forward':
                node_out = np.log(1 + np.exp(node_temp))
            elif direction == 'backward':
                node_out = np.divide(1, np.add(
                    1, np.exp(np.negative(node_temp))))
        else:
            print("activation function is wrong!")
            return 0
        return node_out

