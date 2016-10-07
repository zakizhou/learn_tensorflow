#!/usr/bin/python
"""
uasge : python evaluate.py [model_num]
model_sum should be one of those numbers:{200,400,600,800,1000}
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys

model_num = sys.argv[1]
mnist = input_data.read_data_sets("/home/zhoujie/TensorFlow/MNIST_data/",one_hot=True)

graph = tf.get_default_graph()

with tf.Session() as sess:
    restorer = tf.train.import_meta_graph("/home/zhoujie/TensorFlow/scripts/how_to/import_and_export_graph/proto_buffer/my-model-"+str(model_num)+".meta")
    restorer.restore(sess,"/home/zhoujie/TensorFlow/scripts/how_to/import_and_export_graph/proto_buffer/my-model-"+str(model_num))

    input_placeholder = graph.get_tensor_by_name("inputs:0")
    output_placeholder = graph.get_tensor_by_name("labels:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    print(sess.run(accuracy,feed_dict={input_placeholder:mnist.test.images,output_placeholder:mnist.test.labels}))
