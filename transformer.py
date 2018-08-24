
from sklearn.base import TransformerMixin
from sklearn.linear_model import RidgeClassifier

from sklearn.naive_bayes import GaussianNB


class RidgeTransformer(RidgeClassifier, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)


class GaussianNBTransformer(GaussianNB, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)