"""File which will generate some toy data."""

import pandas as pd
from sklearn.datasets import make_classification


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, n_features=5, n_classes=3,
                               n_informative=4, n_redundant=1, n_repeated=0,
                               random_state=0)
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e'])
    df['class'] = y
    df.to_csv('public_train.csv', index=False)
