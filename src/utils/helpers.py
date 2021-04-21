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
    optimiser_name: str,
    layers: list,
    lr: float,
    weight_decay: float,
    momentum: float,
    exponential_decay: float,
    beta1: float,
    beta2: float,
):

    """
    Creates a optimiser and returns it
    Args:
        optimiser_name (str): Name of optimiser
        layers (list): list of network layer tuples (n_in, n_out)
        lr (float): learning rate
        weight_decay (float): weight decay factor
        momentum (float): momentum factor for SGD
        exponential_decay (float): exponential decay rate for adadelta
        beta1 (float): first smoothing param for adam
        beta2 (float): second smoothing param for adam
    Returns:
        Optimiser object

    """

    if optimiser_name.lower() == "sgd":
        return SGD(layers, lr, weight_decay, momentum)
    elif optimiser_name.lower() == "adadelta":
        return Adadelta(layers, lr, weight_decay, exponential_decay)
    elif optimiser_name.lower() == "adagrad":
        return Adagrad(layers, lr, weight_decay)
    elif optimiser_name.lower() == "adam":
        return Adam(layers, lr, weight_decay, beta1, beta2)
    else:
        raise ValueError(f"Invalid optimiser name: {optimiser_name}")
