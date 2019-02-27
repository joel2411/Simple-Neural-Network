import numpy as np

class Optimizer:
    def __init__(self, optimizer, learning_rate,
                 param1, param2):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.param1 = param1
        self.param2 = param2
        self.moment1 = 0
        self.moment2 = 0
        self.moment3 = 0
        self.moment4 = 0

    def update_layer(self, weight, bias, grad_w, gradient, iter):
        bias = np.subtract(bias,
                           np.multiply(float(self.learning_rate),
                                       np. multiply(gradient, bias)))
        if self.optimizer == 'SGD':
            weight = np.subtract(weight,
                                 float(self.learning_rate) * grad_w)
        elif self.optimizer == 'RMSProp':
            self.moment1 = np.add(self.param1 * self.moment1,
                                  np.multiply(1 - self.param1,
                                              np.multiply(grad_w, grad_w)))
            weight = np.subtract(weight,
                                 np.multiply(
                                     np.divide(grad_w,
                                               np.sqrt(self.moment1) + 1e-7),
                                     float(self.learning_rate)))
        elif self.optimizer == 'SGD_momentum':
            self.moment1 = np.add(self.param1 * self.moment1, grad_w)
            weight = np.subtract(weight,
                                 float(self.learning_rate) * self.moment1)
        elif self.optimizer == 'Nesterov':
            self.moment2 = self.moment1
            self.moment1 = np.subtract(self.param1 * self.moment1,
                                  float(self.learning_rate) * grad_w)
            weight = np.add(np.subtract(weight,
                                        self.param1 * self.moment2),
                            (1 + self.param1) * self.moment1)
        elif self.optimizer == 'AdaGrad':
            self.moment1 = np.add(self.moment1, np.multiply(grad_w, grad_w))
            weight = np.subtract(weight,
                                 np.multiply(
                                     np.divide(grad_w,
                                               np.sqrt(self.moment1) + 1e-7),
                                     float(self.learning_rate)))
        elif self.optimizer == 'Adam':
            self.moment1 = np.add(self.param1 * self.moment1,
                                  np.multiply(1 - self.param1, grad_w))
            self.moment2 = np.add(self.param2 * self.moment2,
                                  np.multiply(1 - self.param2,
                                              np.multiply(grad_w, grad_w)))
            self.moment3 = np.divide(self.moment1,
                                     1 - (self.param1 ** iter))
            self.moment4 = np.divide(self.moment2,
                                     1 - (self.param2 ** iter))
            weight = np.subtract(weight,
                                 np.multiply(
                                     np.divide(self.moment3,
                                               np.sqrt(self.moment4) + 1e-7),
                                     float(self.learning_rate)))
        elif self.optimizer == 'AdaDelta':
            self.moment3 = np.zeros_like(weight)
            self.moment1 = np.add(self.param1 * self.moment1,
                                  np.multiply(1 - self.param1,
                                              np.multiply(grad_w, grad_w)))
            self.moment2 = np.multiply(np.divide(np.sqrt(self.moment3) + 1e7,
                                                 np.sqrt(self.moment1) + 1e7),
                                       grad_w)
            self.moment3 = np.add(self.param2 * self.moment3,
                                  np.multiply(1 - self.param2,
                                              np.power(self.moment2, 2)))
            weight = np.subtract(weight, self.moment2)
        else:
            print("optimizer is wrong")
            return 0
        return weight, bias