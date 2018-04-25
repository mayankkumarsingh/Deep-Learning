import numpy as np

def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """

    # Element-wise exponential
    x_exp = np.exp(x)

    # Vector x_sum contains sums each row of x_exp.
    x_sum = np.sum(x_exp,axis=1,keepdims=True)

    # Compute softmax(x) as x_exp / x_sum
    s = x_exp/x_sum

    return s
