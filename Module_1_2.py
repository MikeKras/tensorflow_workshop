# Tensorflow workshop with Jan Idziak
#-------------------------------------
#
# Based on the tf presentation	 
# First tensor flow 
#
import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(2, 2))
#x**2 + x
y = tf.add(tf.multiply(x, x), x)

rand_array_2 = np.random.rand(2, 2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
result = sess.run(y, feed_dict ={x: rand_array_2})
print(result)

