# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
# Based on the tf presentation	 
# First tensor flow
import numpy as np
import tensorflow as tf

x = tf.placeholder('float', [1,3])
w = tf.Variable(tf.random_normal([3, 3]), name='w')
y = tf.matmul(x, w)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
result = sess.run(y, feed_dict ={x:np.array([[1.0, 2.0, 3.0]])})

print(result)