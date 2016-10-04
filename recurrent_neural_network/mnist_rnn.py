#!/usr/bin/python
# author : windows98@ruc.edu.cn
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/zhoujie/TensorFlow/MNIST_data/",one_hot=True)

input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,784])
labels_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,10])

reshaped_input = tf.reshape(input_placeholder,shape=[-1,28,28])
inputs = [tf.squeeze(input_,[1]) for input_ in tf.split(1,28,reshaped_input)]

cell = rnn_cell.BasicLSTMCell(80,forget_bias=1.0,state_is_tuple=True)

outputs,state = rnn.rnn(cell,inputs,dtype=tf.float32)

Weights = tf.Variable(tf.random_normal(shape=[80,10]))
bias = tf.Variable(tf.random_normal([10]))
logits = tf.nn.xw_plus_b(outputs[-1],Weights,bias)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,labels_placeholder),name="cross_entropy")
predition = tf.equal(tf.argmax(logits,1),tf.argmax(labels_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(predition,tf.float32))

train_step = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        batch_x,batch_y = mnist.train.next_batch(100)
        _,loss = sess.run([train_step,cross_entropy],feed_dict={input_placeholder:batch_x,labels_placeholder:batch_y})
        if((i+1)%20 == 0):
             print("step:"+str(i+1)+" loss: "+str(loss))
        if((i+1)%100 == 0):
            accuracy_this_time = sess.run(accuracy,feed_dict={input_placeholder:mnist.validation.images,labels_placeholder:mnist.validation.labels})
            print(accuracy_this_time)
