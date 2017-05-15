# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
#How to define and operate on matrices (2D tensors)
#diag, random_uniform, convert_to_tensor
#add, transpose, matmul,...

import tensorflow as tf
import numpy as np
sess = tf.Session()

# Create constant_matrix with dim n_row = 3, ncol = 2 
#Constant
a = tf.constant([[1., 2., 3.], [-3., -7., -1.]])
print(sess.run(a))
b = 4*tf.eye(3)
print(sess.run(b))
c = tf.fill([2, 3], 7.0)
print(sess.run(c))
d = tf.subtract(tf.matmul(a, b), tf.exp(c))
print(sess.run(d))
e = tf.log(tf.abs(d))
print(sess.run(e))
# Create rn_matrix with random_normal values with dim nrow = 2, n_col = 3
f = tf.greater(e, 0.0)
print(sess.run(f))
# Create diag_matrix dim 3x3 with 4 on the diagonal
