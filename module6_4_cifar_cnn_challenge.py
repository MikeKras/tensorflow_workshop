# Module 5: Convolutional Neural Network (CNN)
# CNN model with dropout for MNIST dataset

# CNN structure:
# · · · · · · · · · ·      input data                           X  [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x1x6 stride 1         W1 [5, 5, 1, 6]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6x12  stride 2       W2 [5, 5, 6, 12]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer 4x4x12x24 stride 2       W3 [3, 3, 12, 24]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 24]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*24, 200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]
#        · · ·                                                  Y [batch, 10]

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
learning_rate = 0.01
training_epochs = 2
batch_size = 100

import tensorflow as tf

from tflearn.datasets import cifar10
from tflearn.data_utils import to_categorical
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
L1 = 8  # first convolutional layer output depth
L2 = 16  # second convolutional layer output depth
L3 = 32  # third convolutional layer
L4 = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 3, L1], stddev=0.1))
B1 = tf.Variable(tf.zeros([L1]))
W2 = tf.Variable(tf.truncated_normal([5, 5, L1, L2], stddev=0.1))
B2 = tf.Variable(tf.zeros([L2]))
W3 = tf.Variable(tf.truncated_normal([3, 3, L2, L3], stddev=0.1))
B3 = tf.Variable(tf.zeros([L3]))
W4 = tf.Variable(tf.truncated_normal([8 * 8 * L3, L4], stddev=0.1))
B4 = tf.Variable(tf.zeros([L4]))
W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# Step 2: Setup Model
# output is 32x32
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
stride = 2  # output is 16x16
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
Y2 = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME')
stride = 2  # output is 8x8
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 1, 1, 1], padding='SAME') + B3)
Y3 = tf.nn.max_pool(Y3, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME')

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 8 * 8 * L3])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))

# Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    epoch_loss = 0
    for step in range(int(X_train.shape[0] / batch_size)):
        batch_X = X_train[(step*batch_size):((step+1)*batch_size)]
        batch_y = Y_train[(step * batch_size):((step + 1) * batch_size)]
        train_data = {X: batch_X, y: batch_y, pkeep: 0.7}
        _, c = sess.run([train, loss], feed_dict=train_data)
        print("Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))
        epoch_loss += c
    print('Epoch', epoch + 1, 'completed out of', training_epochs, 'loss:', epoch_loss)
acc = []
for i in range(int(X_test.shape[0] / batch_size)):
    acc.append(sess.run(accuracy, {X: X_test[(i*batch_size):((i+1)*batch_size)],
                              y: Y_test[(i*batch_size):((i+1)*batch_size)], pkeep: 1}))
print('Accuracy:', sess.run(tf.reduce_mean(acc)))