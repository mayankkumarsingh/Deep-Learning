

# Test for sigmoid gradient
from sigmoid_gradient import sigmoid_gradient
import numpy
a = numpy.random.randn(2,1)
print(a)
sigmoid_gradient(a)

# Test for sigmoid
from sigmoid import sigmoid
sigmoid(a)
