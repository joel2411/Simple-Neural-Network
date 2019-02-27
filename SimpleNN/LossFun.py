import numpy as np

class Loss:
    def __init__(self, loss_function):
        self.loss_fun = loss_function

    def compute_loss(self, pred, target):
        loss = 0
        if self.loss_fun == 'NLL':
            temp_pred = np.clip(pred, 1e-10, 1 - 1e-10)
            loss = np.where(target == 1,
                            -np.log(temp_pred),
                            -np.log(1 - temp_pred))
        elif self.loss_fun == 'MSE':
            loss = np.average(np.sqrt(np.subtract(pred, target)))
        elif self.loss_fun == 'RMSE':
            loss = np.average(np.abs(np.subtract(pred, target)))
        else:
            print("loss_function is wrong")
            return 0
        return np.sum(loss)

    def compute_grad(self, pred, target):
        if self.loss_fun == 'NLL':
            pred = np.clip(pred, 1e-15, 1 - 1e-15)
            gradient = np.where(target == 1,
                                np.divide(-1, pred),
                                np.divide(1, 1 - pred))
        elif self.loss_fun == 'MSE':
            gradient = np.multiply(2, np.subtract(pred, target))
        elif self.loss_fun == 'RMSE':
            gradient = np.where(pred >= target, 1, -1)
        else:
            print("loss_function is wrong")
            return 0
        return gradient