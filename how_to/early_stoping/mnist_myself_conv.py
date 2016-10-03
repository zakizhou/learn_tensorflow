#!/usr/bin/python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/zhoujie/TensorFlow/MNIST_data/",one_hot=True)

input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,784])
labels_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,10])

reshaped_input = tf.reshape(input_placeholder,[-1,28,28,1])

Weights = {
    "conv_1":tf.Variable(tf.truncated_normal([3,3,1,128],stddev=0.5),dtype=tf.float32),
    "conv_2":tf.Variable(tf.truncated_normal([3,3,128,128],stddev=0.5),dtype=tf.float32)
            }

biases = {
    "conv_1":tf.Variable(tf.zeros([128]),dtype=tf.float32),
    "conv_2":tf.Variable(tf.zeros([128]),dtype=tf.float32)
         }

def conv2d(inputs,weights,bias):
    conv_out = tf.add(tf.nn.conv2d(inputs,weights,strides=[1,1,1,1],padding='SAME'),bias)
    maxpool = tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return tf.nn.relu(maxpool)


conv1_output = conv2d(reshaped_input,Weights["conv_1"],biases["conv_1"])
conv2_output = conv2d(conv1_output,Weights["conv_2"],biases["conv_2"])
reshaped_output = tf.reshape(conv2_output,[-1,7*7*128])

weights_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*128,512],stddev=0.5),dtype=tf.float32)
biases_fc1 = tf.Variable(tf.zeros([512]),dtype=tf.float32)
weights_fc2 = tf.Variable(tf.truncated_normal(shape=[512,10],stddev=0.5),dtype=tf.float32)
biases_fc2 = tf.Variable(tf.zeros([10]),dtype=tf.float32)

hidden1 = tf.nn.relu(tf.nn.xw_plus_b(reshaped_output,weights_fc1,biases_fc1))

logits = tf.nn.xw_plus_b(hidden1,weights_fc2,biases_fc2)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,labels_placeholder))
prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    accuracy_last_time = -1
    for i in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        _,loss = sess.run([train_step,cross_entropy],feed_dict={input_placeholder:batch_x,labels_placeholder:batch_y})
        print("step: "+str(i))
        if((i+1)%100==0):
           accuracy_this_time = sess.run(accuracy,feed_dict={input_placeholder:mnist.validation.images,labels_placeholder:mnist.validation.labels})
           print("accuacy this time:"+str(accuracy_this_time)+" accuracy last time: "+str(accuracy_last_time))
           if(accuracy_this_time > accuracy_last_time):
               print("model saved")
               saver.save(sess,"/home/zhoujie/TensorFlow/scripts/how_to/early_stoping/proto_buffer/model",global_step=(i+1))
               accuracy_last_time = accuracy_this_time
