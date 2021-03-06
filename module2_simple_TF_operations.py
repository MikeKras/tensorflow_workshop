# Module 2: Basic TF Operations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
sess = tf.Session()

# Constants
a = tf.constant(1)
b = tf.constant(2)
c = tf.constant(3)

sess = tf.Session()
print(sess.run(a))
print(sess.run(b))
print(sess.run(c))
sess.close()
with tf.Session() as sess:
	print(sess.run(a))
	print(sess.run(b))
	print(sess.run(c))


sess = tf.Session()
# Math Operations
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
c = tf.mod(c, 7)
print(sess.run(c))
sess.close
# Math Functions
print(sess.run(tf.square(2)))
print(sess.run(tf.sqrt(4.0)))
print(sess.run(tf.sin(3.1416)))
print(sess.run(tf.tan(3.1416)))
print(sess.run(tf.cos(3.1416)))
print(sess.run(tf.exp(1.0)))

# Other Functions
a = tf.linspace(-1., 1., 10)
print(sess.run(a))

# Exercises

a = tf.constant([[1,1]],tf.float32)
w = tf.constant([[1,2],[3,4]],tf.float32)
b = tf.constant([[3.,3.]],tf.float32)
y = tf.add(tf.matmul(a,w),b)
print(sess.run(y))


a = tf.constant([
				[1,2],
				[3,4],
				[5,6]])
print(sess.run(a))

# Matrix
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[4,3],[2,1]])
c = tf.add(a,b)
c = tf.transpose(a)
c = tf.matmul(a,b)
print(sess.run(c))
tf.cross([1, 0, 0], [0, 1, 0])

# Special Matrics
a = tf.zeros([2,3])
a = tf.ones([2,3])
a = tf.diag(np.ones(2))
a = tf.fill([2,3],2)
a = tf.random_uniform([2,3])
print(sess.run(a))

# Random Numbers
tf.set_random_seed(2)
a = tf.random_normal([1])
a = tf.random_uniform([1])
a = tf.random_shuffle([1,2,3,4])
print(sess.run(a))
sess.close()

sess =tf.Session()
# Placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
sum = tf.add(a,b)
print(sess.run(sum,{a:3,b:4}))
print(sess.run(sum,feed_dict={a:3,b:4}))

# Challenge
sess =tf.Session()
x = tf.constant([[1,1]])
w = tf.constant([[1,2],[3,4]])
b = tf.constant([[2, 2]])
y = tf.add(tf.matmul(x,w),b)
print(sess.run(y))
sess.close()

# Challenge
sess =tf.Session()
a = tf.placeholder(tf.float32,shape=[1,2])
w = tf.placeholder(tf.float32,shape=[2,2])
b = tf.constant([[3.,3.]],tf.float32)
y = tf.add(tf.matmul(a,w),b)
print(sess.run(y,feed_dict={
    a:[[1,1]],
    w:[[1,2],[3,4]]})
      )

x = tf.placeholder(tf.float32,shape=(2,2))
y = tf.add(tf.matmul(x,x),x)
print(sess.run(y,feed_dict={x:[[1,1],[1,1]]}))
sess.close()

sess =tf.Session()
# Variables
W = tf.Variable([1.], tf.float32)
b = tf.Variable([-1.], tf.float32)
x = tf.placeholder(tf.float32)
y = W * x + b

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y, {x:[1,2,3,4]}))

