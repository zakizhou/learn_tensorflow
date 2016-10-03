#!/usr/bin/python
# author : windows98@ruc.edu.cn

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/zhoujie/TensorFlow/MNIST_data",one_hot=True)

with tf.Session() as sess:
    restorer = tf.train.import_meta_graph("/home/zhoujie/TensorFlow/scripts/how_to/import_and_export_graph/proto_buffer/my-model.meta")
    restorer.restore(sess,'/home/zhoujie/TensorFlow/scripts/how_to/import_and_export_graph/proto_buffer/my-model')

    input_placeholder = tf.get_collection("input_placeholder")[0]
    output_placeholder = tf.get_collection("output_placeholder")[0]
    cross_entropy = tf.get_collection("cross_entropy")[0]
    accuracy = tf.get_collection("accuracy")[0]

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    accuracy_before_train = sess.run(accuracy,feed_dict={input_placeholder:mnist.test.images,output_placeholder:mnist.test.labels})

    print("accuracy on test set is "+str(accuracy_before_train)+" before train ")

    for i in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={input_placeholder:batch_x,output_placeholder:batch_y})

    accuracy_after_train = sess.run(accuracy,feed_dict={input_placeholder:mnist.test.images,output_placeholder:mnist.test.labels})

    print("accuracy on test set is "+str(accuracy_after_train)+" after train ")

