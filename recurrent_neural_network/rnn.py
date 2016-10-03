#!/usr/bin/python
# author : windows98@ruc.edu.cn
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
import numpy as np
cell = rnn_cell.BasicLSTMCell(10,state_is_tuple=True)
state = cell.zero_state(11,tf.float32)

inputs = tf.constant(np.random.uniform(0,1,(11,12,13)),dtype=tf.float32)

final_inputs = [tf.squeeze(input_,[1]) for input_ in tf.split(1,12,inputs)]

outputs,final_state = rnn.rnn(cell,final_inputs,dtype=tf.float32)


