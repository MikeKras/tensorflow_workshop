# Module 5: Neural Network and Deep Learning
# NN model for MNIST dataset.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.001
training_epochs = 2
batch_size = 100

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

L1 = 200
L2 = 100
L3 = 60
L4 = 30

W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
B1 = tf.Variable(tf.zeros([L1]))
W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
B2 = tf.Variable(tf.zeros([L2]))
W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
B3 = tf.Variable(tf.zeros([L3]))
W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
B4 = tf.Variable(tf.zeros([L4]))
W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# Step 2: Setup Model
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y)

# Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    for i in range(int(55000/batch_size)):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        train_data = {X: batch_X, y: batch_y}
        sess.run(train, feed_dict=train_data)
        print("Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))

# Step 6: Evaluation
test_data = {X:mnist.test.images,y:mnist.test.labels}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))