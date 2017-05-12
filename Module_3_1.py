# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
#script based on the:
# Implementation of a simple MLP network with 
# one hidden layer.
#
# Logistic Regression
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve logistic regression.
# y = argmax(sigmoid(Ax + b))
# y = argmax(sigmoid(Wx))
#
# We will use the iris data, specifically:
#  y = Iris Type 
#  x = Pedal Length, Petal Width, Sepal Width, Sepal Length
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1):
    """
    Forward-propagation.
    """
    yhat = tf.matmul(X, w_1)  # The \sigma function
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris = datasets.load_iris()
    data = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def train_regression():
    train_X, test_X, train_y, test_y = get_iris_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, y_size))

    # Forward propagation
    yhat = forwardprop(X, w_1)
    predict = tf.argmax(yhat, axis=1)
    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(150):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

train_regression()

# Exercise for module 3_1 
# Prepare script for linear regression for the Iris data
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Length, Petal Width, Sepal Width

# Use functions
# - init_weights,
# - forwardprop -> modify for linear regression, 
# - get_iris_data -> modify for data entry train/target