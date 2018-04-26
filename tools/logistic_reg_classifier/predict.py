import numpy as np
from tools.basic_functions.sigmoid import sigmoid
# import sys
# sys.path.insert(0,"../basic_functions")
# from sigmoid import sigmoid
# sys.path.pop(0)


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    mtest = X.shape[1]
    Y_prediction = np.zeros((1,mtest))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction[0,i] = np.round(A[0,i])

    assert(Y_prediction.shape == (1, mtest))

    return Y_prediction
