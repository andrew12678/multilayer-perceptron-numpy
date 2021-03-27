from abc import ABCMeta, abstractmethod


class Activation(metaclass=ABCMeta):
    """
    The base class for an activation
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        return

    @abstractmethod
    def backward(self, upstream_grad):
        return

    @abstractmethod
    def derivative(self, x):
        return
