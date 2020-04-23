from ..utils import shuffle

def split_cv(X_in, y_in, ratio=0.3):
    """
    Randomly split data into training and crossvalidation sets using given ratio

    Input:
    ---
    X_in : Matrix
        a
    y_in : Vector
        a
    ratio : Number
        The ratio of the input that should be split into a crossvalidation set.


    Output:
    ---
    X_train : Matrix
    y_train : Vector
    X_cv : Matrix
    y_cv : Vector
    """
    X, y = shuffle(X_in, y_in)
    bp = int((1-ratio) * len(y))
    X_train = X[:bp]
    y_train = y[:bp]
    X_cv = X[bp:]
    y_cv = y[bp:]
    return X_train, y_train, X_cv, y_cv
