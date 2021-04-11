# comp5329-assignment1

Welcome to our assignment, we made it our goal to have a clear OOP setup for our implemention of the MLP.
 
For experimentation, we have implemented YAML files to change configurations.


## Dependencies

Python Version: `python=3.8`

Dependencies are stored in `requirements.txt`. 

If you install new packages please use `pip list --format freeze > requirements.txt` to replace the file.

Code is auto-formatted using `black`: installation instructions for IDEs: https://black.readthedocs.io/en/stable/editor_integration.html

Docstrings are Google style  

## Folder Structure

```
ROOT/
├── hyperparams/ - YAML files for our experiment config
│ └── config.yml
├── main.py - main file for running our experiments
├── requirements.txt - requirements for our project
└── src/ - source code for our MLP implementation
    ├── activation/ - directory for our activations
    │ ├── activation.py - base case for activation
    │ ├── identity.py - identity activation function
    │ ├── leaky_relu.py - leaky ReLU activation function
    │ ├── logistic.py - logistic activation function
    │ ├── relu.py - ReLU activation function
    │ └── tanh.py - tanh activation function
    ├── layers/ - directory for our layers
    │ ├── layer.py - base class for layers
    │ ├── batch_norm.py - BatchNorm layer
    │ └── linear.py - Linear layer
    ├── loss/ - directory for our loss functions
    │ ├── loss.py - base class for loss functions
    │ ├── cross_entropy.py - cross entropy loss
    │ └── mean_square_error.py - MSE loss
    ├── network/ - directory for our NN
    │ └── mlp.py - MLP class
    ├── optimiser/ - directory for our optimisers
    │ ├── optimiser.py - base class for optimisers
    │ ├── adagrad.py - AgaGrad optimiser
    │ ├── adam.py - ADAM optimiser
    │ ├── adadelta.py - AdaDelta optimiser
    │ └── sgd.py - SGD optimiser (handles momentum interally)
    ├── trainer/ - directory for our Trainer object
    │ └── trainer.py
    └── utils/ - misc utils 
        ├── calculations.py - mainly mathematical computations
        ├── helpers.py - object initialisers
        ├── io.py - input output
        ├── metrics.py - ML metrics 
        └── ml.py - other ML related tasks

  ```

  