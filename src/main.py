import numpy as np
import openjij as oj
from mnist_ml import MNistML
from surrogate_model import BOCSSurrogateModel
from acquisition_function import AcquisitionFunction

def save_optimizing_data():
    pass

def show_figure(data, title):
    pass


if __name__ == "__main__":
    # set up
    D = 1
    iter_init = 5
    iter_max = 20
    ml_model = MNistML()
    bbo_model = BOCSSurrogateModel(D)
    acq_func = AcquisitionFunction()

    # init dataset
    X = np.random.rand(iter_init, D)
    y = ml_model.predict(X)

    print(X)
    print(y)

    # init surrogate model
    bbo_model.fit(X, y)

    # init acquisition function
    bbo_params = bbo_model.params
    acq_func.build(bbo_params)
    next_x = acq_func.optimize()

    # update dataset
    X = np.vstack((X, next_x))
    y_new = ml_model.predict(next_x)
    y = np.hstack((y, y_new))

    # iterate BOCS
    for iter in range(iter_max):
        # update surrogate model
        bbo_model.fit(X, y)

        # update acquisition function
        bbo_params = bbo_model.params
        acq_func.build(bbo_params)
        next_x = acq_func.optimize()

        # update dataset
        X = np.vstack((X, next_x))
        y_new = ml_model.predict(next_x)
        y = np.hstack((y, y_new))

        save_optimizing_data()

    # output results
