#!/usr/bin/python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/zhoujie/TensorFlow/MNIST_data/",one_hot=True)

input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,784],name="inputs")
output_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,10],name="labels")

Weights = tf.Variable(tf.truncated_normal(shape=[784,10],stddev=0.5),dtype=tf.float32)
biases = tf.Variable(tf.zeros([10]),dtype=tf.float32)

logits = tf.nn.xw_plus_b(input_placeholder,Weights,biases)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,output_placeholder),name="loss")
prediction = tf.equal(tf.argmax(logits,1),tf.argmax(output_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32),name="accuracy")

init = tf.initialize_all_variables()

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#tf.add_to_collection("cross_entropy",cross_entropy)
#tf.add_to_collection("input_placeholder",input_placeholder)
#tf.add_to_collection("output_placeholder",output_placeholder)
#tf.add_to_collection("accuracy",accuracy)

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={input_placeholder:batch_x,output_placeholder:batch_y})

        saver = tf.train.Saver()
        if((i+1)%200 == 0):
            print("save :"+str((i+1)/200))
            saver.save(sess,"/home/zhoujie/TensorFlow/scripts/how_to/import_and_export_graph/proto_buffer/my-model",global_step=(i+1))
            #saver.export_meta_graph("/home/zhoujie/TensorFlow/scripts/how_to/import_and_export_graph/proto_buffer/my-model.meta")


