# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
#How to define and operate on matrices (2D tensors)
#diag, random_uniform, convert_to_tensor
#add, transpose, matmul,...

import tensorflow as tf
import numpy as np
sess = tf.Session()

#Declaration of matrices
tensor = tf.zeros([1, 10])
print(tensor)

constant = tf.constant([2, 3])
print(constant)

#Identity
identity_matrix = tf.diag(np.ones(3))
identity_matrix = tf.cast(identity_matrix,dtype='float')
print(sess.run(identity_matrix))

#Constant
const_matrix = tf.fill([2, 3], -1.0)
print(sess.run(const_matrix))

#Converted
converted_matrix =  tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(converted_matrix))

# Add two matrices
print(sess.run(tf.add(identity_matrix, identity_matrix)))
print(sess.run(tf.subtract(identity_matrix, identity_matrix)))

#Matrix multiplication and transpose
uni_times_const = tf.matmul(const_matrix, identity_matrix)
print(sess.run((uni_times_const)))

#Inverse
two_identity = tf.add(identity_matrix, identity_matrix)
print(sess.run(tf.matrix_inverse(two_identity)))

### Exercise modelue_1_2

# Create constant_matrix with dim n_row = 3, ncol = 2 
#Constant
const_matrix = tf.fill([2, 3], 50.0)
print(sess.run(const_matrix))
# Create rn_matrix with random_normal values with dim nrow = 2, n_col = 3
normal_matrix = tf.random_normal([3,2]) 
print(sess.run(normal_matrix))
# Create diag_matrix dim 3x3 with 4 on the diagonal
diag_matrix = 4*tf.diag(np.ones(3))
print(sess.run(diag_matrix))
# Multiply constant matrix and rn_matrix
mm = tf.matmul(const_matrix, normal_matrix)
print(sess.run(mm))
# Add diag_matrix
# Print results