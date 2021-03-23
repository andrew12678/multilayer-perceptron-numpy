import numpy as np

import utils
from mlp import MLP


def run():
    X_train, y_train, X_test, y_test = utils.load_directory("data")

    # splits = utils.create_kfold_stratified_cross_validation(
    # X_train, y_train, X_test, y_test, 10
    # )

    num_features = X_train.shape[1]
    classes = np.unique(y_train)

    nn = MLP([num_features, 10, len(classes)], [None, "logistic", "tanh"])

    input_data = X_train[:1000, :]
    output_data = y_train[:1000, :]

    ### Try different learning rate and epochs
    MSE = nn.fit(input_data, output_data, learning_rate=0.001, epochs=50)
    print("loss:%f" % MSE[-1])


if __name__ == "__main__":
    run()
