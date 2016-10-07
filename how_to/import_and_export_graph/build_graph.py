#!/usr/bin/python
# author : windows98@ruc.edu.cn
"""
This is the script used to build a graph within in which a softmax regression model is trained to fit the
mnist data and then save the graph(model) to the my-model.meta and then restore this model in the import_graph.py.
To run this script,just use 'python build_graph.py'.
"""

import tensorflow as tf

input_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,784],name="inputs")
output_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,10],name="labels")

Weights = tf.Variable(tf.truncated_normal(shape=[784,10],stddev=0.5),dtype=tf.float32)
biases = tf.Variable(tf.zeros([10]),dtype=tf.float32)

logits = tf.nn.xw_plus_b(input_placeholder,Weights,biases)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,output_placeholder),name="loss")
prediction = tf.equal(tf.argmax(logits,1),tf.argmax(output_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32),name="accuracy")

init = tf.initialize_all_variables()

#tf.add_to_collection("cross_entropy",cross_entropy)
#tf.add_to_collection("input_placeholder",input_placeholder)
#tf.add_to_collection("output_placeholder",output_placeholder)
#tf.add_to_collection("accuracy",accuracy)

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess,"/home/zhoujie/TensorFlow/scripts/how_to/import_and_export_graph/proto_buffer/my-model")
    saver.export_meta_graph("/home/zhoujie/TensorFlow/scripts/how_to/import_and_export_graph/proto_buffer/my-model.meta")
