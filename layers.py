import numpy as np

from activation import Activation


class HiddenLayer(object):
    def __init__(
        self,
        n_in,
        n_out,
        activation_last_layer="tanh",
        activation="tanh",
        W=None,
        b=None,
    ):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = None
        self.activation = Activation(activation).f

        # activation deriv of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        # we randomly assign small values for the weights as the initiallization
        self.W = np.random.uniform(
            low=-np.sqrt(6.0 / (n_in + n_out)),
            high=np.sqrt(6.0 / (n_in + n_out)),
            size=(n_in, n_out),
        )
        if activation == "logistic":
            self.W *= 4

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(
            n_out,
        )

        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    # the forward and backward progress for each training epoch
    # please learn the week2 lec contents carefully to understand these codes.
    def forward(self, input):
        """
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        """
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None else self.activation(lin_output)
        )
        self.input = input
        return self.output

    def backward(self, delta, output_layer=False):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta
