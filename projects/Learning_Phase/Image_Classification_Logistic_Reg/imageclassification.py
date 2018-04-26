import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from loadDataset import load_dataset
import sys
sys.path.insert(0,"../../../")
sys.path
from tools.logistic_reg_classifier.logistic_reg_model import logistic_reg_model
sys.path.pop(0)

from tools.logistic_reg_classifier import logistic_reg_model



# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, \
    test_set_x_orig, test_set_y, classes = load_dataset()

#######################################################################
####        SECTION: Display Data       ###############################
#######################################################################
## PLOTTING Images from dataset
# Example of a picture
index = 30
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" +
    classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +
     "' picture.")
plt.pause(0.005)

# generation of a dictionary of (title, images)
from plotFigures import plot_figures
number_of_im = 20
w=10
h=10
figures = {'Image: '+ str(i):train_set_x_orig[np.random.randint(1,train_set_x_orig.shape[0])] for i in range(number_of_im)}
plot_figures(figures, 5, 4)
plt.pause(0.005)

######################    END OF SECTION    ###########################



#######################################################################
####        SECTION: Calculate Parameters       #######################
#######################################################################
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

######################    END OF SECTION    ###########################


#######################################################################
####        SECTION: PreProcessing Data       #########################
#######################################################################
# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

##  Normalizing Colors to 0 - 1
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

######################    END OF SECTION    ###########################


#######################################################################
####        SECTION: Loading Required ML Functions       ##############
#######################################################################

# Functions defined in mayankMLLib.py
# Sanity Check of Imported Functions
        # print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
        #
        #
        # del w
        # del b
        # dim = 2
        # w, b = initialize_with_zeros(2)
        # print ("w = " + str(w))
        # print ("b = " + str(b))
        #
        # del w, b
        # w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
        # grads, cost = propagate(w, b, X, Y)
        # print ("dw = " + str(grads["dw"]))
        # print ("db = " + str(grads["db"]))
        # print ("cost = " + str(cost))
        #
        #
        # params, grads, costs = gradientDescent(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
        #
        # print ("w = " + str(params["w"]))
        # print ("b = " + str(params["b"]))
        # print ("dw = " + str(grads["dw"]))
        # print ("db = " + str(grads["db"]))
        #
        # w = np.array([[0.1124579],[0.23106775]])
        # b = -0.3
        # X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
        # print ("predictions = " + str(predict(w, b, X)))

d = logistic_reg_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()



## END OF PROGRAM
# Use plt.show to prevent figures from closing
plt.show()
