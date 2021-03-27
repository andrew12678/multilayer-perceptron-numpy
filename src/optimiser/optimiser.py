from abc import ABCMeta, abstractmethod


class Optimiser(metaclass=ABCMeta):
    """
    The base class for a layer
    """

    def __init__(self, layers: list):
        self.layers = layers

    @abstractmethod
    def step(self, *args, **kwargs):
        return
