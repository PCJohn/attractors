from __future__ import division
import tensorflow as tf

relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax
tanh = tf.nn.tanh

xinit = tf.contrib.layers.xavier_initializer()
binit = tf.constant_initializer(0.0)

def dense(inp,size,act=None,name=None):
    outp = tf.layers.dense(inp,size,kernel_initializer=xinit,bias_initializer=binit,name=name)
    if act is not None:
        return act(outp)
    return outp

def layer_norm(inp,gain,bias):
    mean,var = tf.nn.moments(inp,1)
    inp = tf.squeeze(inp,-1)
    std = tf.sqrt(var)
    norm = (inp-mean) / (std + 1e-5)
    return tf.expand_dims(gain * norm + bias, -1)

def outer(a,b):
    return tf.matmul(tf.expand_dims(a,-1),tf.transpose(tf.expand_dims(b,-1),[0,2,1]))


