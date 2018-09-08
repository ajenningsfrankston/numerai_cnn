
from sklearn.linear_model import RidgeClassifier
import numpy as np


class PRidgeClassifier(RidgeClassifier):

    def predict_proba(self, X):
#
        d = self.decision_function(X)
        prob = np.exp(d)/(1+np.exp(d))
        pdv = np.array([1-prob, prob])
        pdv = pdv.T
        return pdv
