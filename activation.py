import numpy as np


class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a ** 2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x)
        return a * (1 - a)

    def __init__(self, activation="tanh"):
        if activation == "logistic":
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == "tanh":
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
