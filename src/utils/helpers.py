from ..activation import *
from ..loss import *
from ..optimiser import *


def create_activation_layer(activation_name: str):
    """
    Creates an activation layer and returns it
    Args:
        activation_name (str): Name of activation function

    Returns:

    """
    if activation_name.lower() == "relu":
        return ReLU()
    elif activation_name.lower() == "leaky_relu":
        return LeakyReLU()
    elif activation_name.lower() in {"sigmoid", "logistic"}:
        return Logistic()
    elif activation_name.lower() == "tanh":
        return Tanh()
    elif activation_name.lower() == "identity":
        return Identity()
    else:
        raise ValueError(f"Invalid activation name: {activation_name}")


def create_loss_function(loss_name: str):
    """
    Creates a loss function and returns it
    Args:
        loss_name (str): Name of loss function

    Returns:

    """
    if loss_name == "cross_entropy":
        return CrossEntropyLoss()
    elif loss_name == "mse":
        return MeanSquaredErrorLoss()
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")


def create_optimiser(
    optimiser_name: str, layers: list, lr: int, weight_decay: float, momentum: float
):
    """
    Creates a optimiser and returns it
    Args:
        optimiser_name (str): Name of optimiser

    Returns:

    """
    if optimiser_name == "sgd":
        return SGD(layers, lr, weight_decay, momentum)
