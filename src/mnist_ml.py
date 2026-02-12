import numpy as np
class MNistML:
    def __init__(self):
        self.tuned_hyperparams = None

    def predict(self):
        return np.sum(self.tuned_hyperparams, axis=1)
    
    def loss(self):
        return self.predict()
    
    def fit(self, X):
        self.tuned_hyperparams = X
        loss = self.loss()
        return loss