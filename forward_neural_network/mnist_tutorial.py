#!/usr/bin/python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#read datasets
mnist = input_data.read_data_sets("/home/zhoujie/TensorFlow/MNIST_data/",one_hot=True)

sess = tf.InteractiveSession()

weights = tf.Variable(tf.zeros([784,10]))
biases = tf.Variable(tf.zeros([10]))

input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,784])
output_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,10])

logits = tf.add(tf.matmul(input_placeholder,weights),biases)
prediction = tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_placeholder*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.initialize_all_variables().run()
for i in range(1000):
    batch_x,batch_y = mnist.train.next_batch(100)
    train_step.run({input_placeholder:batch_x,output_placeholder:batch_y})

correction_prediction = tf.equal(tf.argmax(output_placeholder,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

print(accuracy.eval({input_placeholder:mnist.test.images,output_placeholder:mnist.test.labels}))

