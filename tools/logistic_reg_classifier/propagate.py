# import sys
# sys.path.insert(0,"../basic_functions")
# from sigmoid import sigmoid
# sys.path.pop()
from tools.basic_functions.sigmoid import sigmoid
import numpy as np
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # compute activation
    A = sigmoid(np.dot(w.T, X) + b)
    # compute cost
    J = -(np.dot(Y, np.log(A.T)) + np.dot((1-Y), np.log(1-A.T)))/m

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw =  1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    J = np.squeeze(J)
    assert(J.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, J
