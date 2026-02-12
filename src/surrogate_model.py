import numpy as np
from horseshoe_distribution import HorseshoeDistribution

def calc_p_from_d(d):
    p = 1 + d + d * (d - 1) // 2
    return p

def calc_z_from_x(x):
    d = len(x)
    p = calc_p_from_d(d)

    z = np.zeros(p, dtype=int)
    z[0] = 0
    z[1:d+1] = x
    # 上三角行列(対角成分はゼロ)
    i, j = np.triu_indices(d, k=1)
    z[d+1:] = x[i] * x[j]
    return z

class BOCSSurrogateModel:
    def __init__(self, D):
        """
        Class of the surrogate model for BOCS.
        Args:
            D (int): Dimension of the input space
        """
        p = calc_p_from_d(D)
        self.linear_regression_model = HorseshoeDistribution(p)

    def init_fit(self, X, Y):
        """
        Initial fit of the surrogate model with the given dataset.
        Args:
            X (numpy.ndarray): Input data of shape (N, D)
            Y (numpy.ndarray): Output data of shape (N,)
        Returns:
            None
        """
        self.linear_regression_model.init_fit(X, Y)
        
    def fit(self, X, Y):
        """
        Update the surrogate model with the given dataset.
        Args:
            X (numpy.ndarray): Input data of shape (N, D)
            Y (numpy.ndarray): Output data of shape (N,)
        Returns:
            None
        """

        self.linear_regression_model.fit(X, Y)