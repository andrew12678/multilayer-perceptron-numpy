# comp5329-assignment1

Welcome to our assignment, we made it our goal to have a clear OOP setup for our implemention of the MLP.
 
For experimentation, we have implemented YAML files to change configurations.

The general model design can be seen below (for the assumed 128-input and 10-output dimensions):

![Model Architecture](Model_Architecture.png)


## Dependencies

Python Version: `python=3.8`

Dependencies are stored in `requirements.txt`. 

If you install new packages please use `pip list --format freeze > requirements.txt` to replace the file.

Code is auto-formatted using `black`: installation instructions for IDEs: https://black.readthedocs.io/en/stable/editor_integration.html

Docstrings are Google style  

## Running the code
Note that the below commands will run the best model, which corresponds to _params-best_ in _hyperparams/config.yml_. To run another model, replace _params-best_ with another model name in _hyperparams/config.yml_.

### Prepare Data

Please import the following files to the "data" folder in the root directory:
1. test_data.npy 
2. test_label.npy 
3. train_data.npy 
4. train_label.npy 

#### To run the best model on the entire training set and test on the test set, run:
```
python main.py -hy params-best
```
The above command takes ~22 minutes on a machine with 12 cores.

#### To create learning curves for the best model, run:
```
python main.py -lc 1 -hy params-best
```
#### To run the complete ablation studies for the best model, run:
```
python main.py -a 1 -hy params-best
```
#### For 5-folds stratified cross validation with the best model, run:
```
python main.py -hy params-best -kf 1 -p [number of processors to use]
```
#### To find the best model using a grid-search with 5-folds cross validation, run:
```
python main.py -hy grid-search1 -kf 1 -p [number of processors to use]
```
where _grid-search1_ represents the hyperparameter values to search over, defined in _hyperparams/config.yml_.

#### To plot the best model train and validation errors with respect to the number of epochs, run:
```
python main.py -kf 1 -pe 1 -hy params-best
```
#### To generate a model performance table after a (set of) grid search, run:
```
python analysis/hyperparam_plots.py -rf [set of results files]
```
where [set of results files] can be one or more grid search result json files from _results/_.

## Folder Structure

```
ROOT/
├── analysis
│   ├── ablations
│   ├── hyperparam_plots.py
│   ├── learning_curves
│   └── losses
├── data
│   ├── test_data.npy
│   ├── test_label.npy
│   ├── train_data.npy
│   └── train_label.npy
├── hyperparams
│   └── config.yml
├── main.py
├── Model_Architecture.png
├── README.md
├── requirements.txt
├── results
└── src
    ├── activation
    │   ├── activation.py
    │   ├── __init__.py
    │   ├── leaky_relu.py
    │   ├── logistic.py
    │   ├── relu.py
    │   └── tanh.py
    ├── experiments.py
    ├── layers
    │   ├── batch_norm.py
    │   ├── layer.py
    │   ├── linear.py
    ├── loss
    │   ├── cross_entropy.py
    │   ├── __init__.py
    │   ├── loss.py
    │   ├── mean_square_error.py
    ├── network
    │   ├── mlp.py
    ├── optimiser
    │   ├── adadelta.py
    │   ├── adagrad.py
    │   ├── adam.py
    │   ├── __init__.py
    │   ├── optimiser.py
    │   └── sgd.py
    ├── plotting.py
    ├── trainer
    │   └── trainer.py
    └── utils
        ├── calculations.py
        ├── helpers.py
        ├── io.py
        ├── metrics.py
        ├── ml.py

  ```

  