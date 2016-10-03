#!/usr/bin/python
# author : windows98@ruc.edu.cn

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/zhoujie/TensorFlow/MNIST_data",one_hot=True)

Weights = {"hidden1":tf.Variable(tf.truncated_normal([784,512],stddev=0.5)),
           "hidden2":tf.Variable(tf.truncated_normal([512,10],stddev=0.5))}

biases = {"hidden1":tf.Variable(tf.zeros([512])),
          "hidden2":tf.Variable(tf.zeros([10]))}

input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,784])
output_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,10])

hidden1_output = tf.nn.xw_plus_b(input_placeholder,Weights["hidden1"],biases["hidden1"])
hidden2_input = tf.nn.relu(hidden1_output)
logits = tf.nn.xw_plus_b(hidden2_input,Weights["hidden2"],biases["hidden2"])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,output_placeholder))
correction_prediction = tf.equal(tf.argmax(output_placeholder,1),tf.argmax(logits,1))
accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={input_placeholder:batch_x,output_placeholder:batch_y})

    print(sess.run(accuracy,feed_dict={input_placeholder:mnist.test.images,output_placeholder:mnist.test.labels}))
