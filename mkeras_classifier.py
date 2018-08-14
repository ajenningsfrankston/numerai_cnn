from keras.wrappers.scikit_learn import KerasClassifier

class MKerasClassifier(KerasClassifier):
    """
     Extends KerasClassifier to modify the accuracy_score function
    """

    def predict(self, x):
        """
            modify case of 0.5 as boundary
        """
        score = super().predict(x)
        score.flatten()
        score[score >= 0.5] = 1
        score[score < 0.5] = 0
        return score





