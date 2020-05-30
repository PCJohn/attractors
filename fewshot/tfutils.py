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
    if not (act is None):
        return act(outp)
    return outp

def conv(inp,size,act=None,name=None,width=3,strides=2):
    outp = tf.layers.conv2d(inp,size,width,strides=strides,padding='same',kernel_initializer=xinit,bias_initializer=binit,name=name)
    if not (act is None):
        return act(outp)
    return outp

def conv_wts(w,h,in_ch,out_ch):
    return [tf.Variable(xinit([w,h,in_ch,out_ch]),trainable=True,dtype=tf.float32),
            tf.Variable(tf.zeros([out_ch]),trainable=True,dtype=tf.float32)]

def layer_norm(inp,gain,bias):
    mean,var = tf.nn.moments(inp,1)
    inp = tf.squeeze(inp,-1)
    std = tf.sqrt(var)
    norm = (inp-mean) / (std + 1e-5)
    k_ = tf.expand_dims(gain * norm + bias, -1)
    return k_

def layer_norm_1d(inp,gain,bias):
    mean,var = tf.nn.moments(inp,1)
    mean = tf.expand_dims(mean,-1)
    var = tf.expand_dims(var,-1)
    #import pdb; pdb.set_trace();
    std = tf.sqrt(var)
    norm = (inp-mean) / (std + 1e-5)
    k_ = gain * norm + bias
    #import pdb; pdb.set_trace();
    return k_

def outer(a,b):
    return tf.matmul(tf.expand_dims(a,-1),tf.transpose(tf.expand_dims(b,-1),[0,2,1]))

def batch_matmul(mat,batch_feat):
    def mul_fn(mat):
        return lambda x_:tf.matmul(mat,x_)
    mul = mul_fn(mat)
    prod = tf.map_fn(mul, batch_feat)
    #import pdb; pdb.set_trace();
    return prod

