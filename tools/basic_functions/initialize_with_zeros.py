import numpy as np
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and
    initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim,1))
    b = np.float(0)

    assert(w.shape == (dim, 1)), 'Oh man, size of w is not correct'
    assert(isinstance(b, float) or isinstance(b, int)), 'b aint int or float'

    return w, b
