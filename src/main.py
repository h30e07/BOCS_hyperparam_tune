import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
from mnist_ml import MNistML
from surrogate_model import BOCSSurrogateModel
from acquisition_function import AcquisitionFunction

def save_optimizing_data():
    pass

def show_figure(data, title):
    pass

if __name__ == "__main__":
    # set up
    D = 4
    iter_init = 5
    iter_max = 20
    ml_model = MNistML()
    bbo_model = BOCSSurrogateModel(D)
    acq_func = AcquisitionFunction()

    # init dataset
    X = np.random.randint(0, 2, size=(iter_init, D), dtype=np.int8)
    Y = ml_model.fit(X)

    print(X)
    print(Y)

    # init surrogate model
    bbo_model.init_fit(X, Y)

    # init acquisition function
    bbo_params = bbo_model.params
    acq_func.build(bbo_params)
    x_new = acq_func.optimize()

    # update dataset
    X = np.vstack((X, x_new))
    y_new = ml_model.fit(x_new)
    Y = np.hstack((Y, y_new))

    # iterate BOCS
    for iter in range(iter_max):
        # update surrogate model
        bbo_model.fit(X, Y)

        # update acquisition function
        bbo_params = bbo_model.params
        acq_func.build(bbo_params)
        x_new = acq_func.optimize()

        # update dataset
        X = np.vstack((X, x_new))
        y_new = ml_model.fit(x_new)
        Y = np.hstack((Y, y_new))

        # save data from the optimization process
        save_optimizing_data()

    # output results
