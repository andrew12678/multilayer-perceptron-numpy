from .optimiser import Optimiser
from typing import List
from ..layers.linear import Layer
import numpy as np


class Adam(Optimiser):
    def __init__(
        self,
        layers: List[Layer],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0,
    ):
        """
        Initialise parameters for Adam optimiser. Default values given by the paper: https://arxiv.org/pdf/1412.6980.pdf
        Args:
            layers (Linear[Layer]): List of parameters holding Layer objects (either BN or Linear)
            learning_rate (float): the learning rate of the optimiser
            beta1 (float): decay rate for m_t
            beta2 (float): decay rate for v_t
            weight_decay (float): the decay factor applied to the weights before an update.
        """
        super().__init__(layers)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 10e-8
        self.t = 1
        self.weight_decay = weight_decay

        self.v_t = [
            [np.zeros(layer.grad_W.shape), np.zeros(layer.grad_B.shape)]
            for layer in layers
        ]  # Set this to 0 initially since all exponentially decaying average of past squared gradient are 0

        self.m_t = [
            [np.zeros(layer.grad_W.shape), np.zeros(layer.grad_B.shape)]
            for layer in layers
        ]  # Set this to 0 initially since all exponentially decaying average of past gradients are 0

    def step(self):

        for idx, layer in enumerate(self.layers):

            # Computing exponentially decaying average of past gradients
            new_m_t_W = self.m_t[idx][0] * self.beta1 + (1 - self.beta1) * layer.grad_W
            new_m_t_B = self.m_t[idx][1] * self.beta1 + (1 - self.beta1) * layer.grad_B

            # Computing exponentially decaying average of past gradients squared
            new_v_t_W = self.v_t[idx][0] * self.beta2 + (1 - self.beta2) * (
                layer.grad_W ** 2
            )
            new_v_t_B = self.v_t[idx][1] * self.beta2 + (1 - self.beta2) * (
                layer.grad_B ** 2
            )

            # Update for next time
            self.m_t[idx][0] = new_m_t_W
            self.m_t[idx][1] = new_m_t_B

            self.v_t[idx][0] = new_v_t_W
            self.v_t[idx][1] = new_v_t_B

            # Bias correction
            m_t_W_hat = new_m_t_W / (1 - (self.beta1 ** self.t))
            m_t_B_hat = new_m_t_B / (1 - (self.beta1 ** self.t))

            v_t_W_hat = new_v_t_W / (1 - (self.beta2 ** self.t))
            v_t_B_hat = new_v_t_B / (1 - (self.beta2 ** self.t))

            # Update values
            update_W = self.lr / (np.sqrt(v_t_W_hat) + self.epsilon) * m_t_W_hat
            update_B = self.lr / (np.sqrt(v_t_B_hat) + self.epsilon) * m_t_B_hat

            # Update weights
            layer.weights = (1 - self.weight_decay) * (layer.weights - update_W)

            # Update biases
            layer.biases -= update_B

        self.t += 1
