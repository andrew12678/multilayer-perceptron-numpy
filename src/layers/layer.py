from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    """
    The base class for a layer
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
    def zero_grad(self):
        return
