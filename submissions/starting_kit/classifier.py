from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier


class Classifier(BaseEstimator):
    """Classifier used during the submission.

    This classifier will be trained and then use to solve our problem.

    Parameters
    ----------
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``.

    Attributes
    ----------
    clf_ : object,
        The classifier used internally.

    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        """Train our classifier using training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The training data.

        y : ndarray, shape (n_samples, )
            The target associated with the data.

        Returns
        -------
        self
        """
        # we will use a dummy classifier
        self.clf_ = DummyClassifier(random_state=self.random_state)
        self.clf_.fit(X, y)

        return self

    def predict(self, X):
        """Predict target providing test data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        pred : ndarray, shape (n_samples, )
            Prediction labels.

        """
        return self.clf_.predict(X)

    def predict_proba(self, X):
        """Predict target providing test data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        pred : ndarray, shape (n_samples, n_classes)
            Prediction probabilities.

        """
        return self.clf_.predict_proba(X)
