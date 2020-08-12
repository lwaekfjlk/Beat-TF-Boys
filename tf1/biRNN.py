import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

feature_input = 28
timesteps = 28
feature_hidden = 128
num_classes = 10
# X size  (batch_size, sample_number, feature_number)
X = tf.placeholder("float", [None, timesteps, feature_input])
Y = tf.placeholder("float", [None, num_classes])
weights = {'out' : tf.Variable(tf.random_normal([2*feature_hidden,num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

def BiRNN(x, weights, biases):
    x = tf.unstack(x,num=timesteps,axis=1)

    lstm_fw_cell = rnn.BasicLSTMCell(feature_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(feature_hidden, forget_bias=1.0)

    outputs,_,_ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    sess.run(init)
    batch_size = 128
    for step in range(1, 10000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, feature_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if (step % 200 == 0):
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("step : {} loss : {} acc : {}".format(step, loss, acc))

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, feature_input))
    test_label = mnist.test.labels[:test_len]
    print("test acc : {}".format(sess.run(accuracy, feed_dict={X:test_data,Y:test_label})))

