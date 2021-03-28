from abc import ABCMeta, abstractmethod


class Loss(metaclass=ABCMeta):
    """
    The base class for a loss
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
