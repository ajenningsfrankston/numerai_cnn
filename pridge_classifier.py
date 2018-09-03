
from sklearn.linear_model import RidgeClassifier
import numpy as np


class PRidgeClassifier(RidgeClassifier):

    def _predict_proba_(self, X):
#
        d = self.decision_function(X)
        prob = np.exp(d)/(1+np.exp(d))
        return prob
