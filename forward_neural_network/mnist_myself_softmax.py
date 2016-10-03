#!/usr/bin/python
# author : windows98@ruc.edu.cn

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/zhoujie/TensorFlow/MNIST_data/",one_hot=True)

Weights = tf.Variable(tf.zeros([784,10]))
biases = tf.Variable(tf.zeros([10]))

input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,784])
output_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,10])
logits = tf.add(tf.matmul(input_placeholder,Weights),biases)

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,tf.argmax(output_placeholder,1)))

correct_prediction = tf.equal(tf.argmax(output_placeholder,1),tf.argmax(logits,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={input_placeholder:batch_x,output_placeholder:batch_y})

    print(sess.run(accuracy,feed_dict={input_placeholder:mnist.test.images,output_placeholder:mnist.test.labels}))
