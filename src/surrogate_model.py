import numpy as np

class BOCSSurrogateModel:
    def __init__(self, D):
        self.params = np.random.rand(D)

    def fit(self, X, y):
        pass