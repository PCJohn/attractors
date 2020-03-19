import tensorflow as tf

relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax
tanh = tf.nn.tanh

xinit = tf.contrib.layers.xavier_initializer()
binit = tf.constant_initializer(0.0)

def dense(inp,size,act=None):
    outp = tf.layers.dense(inp,size,kernel_initializer=xinit,bias_initializer=binit)
    if act is not None:
        return act(outp)
    return outp

def layer_norm(inp):
    return tf.contrib.layers.layer_norm(inp,center=True,scale=True)

def outer(a,b):
    return tf.matmul(tf.expand_dims(a,-1),tf.transpose(tf.expand_dims(b,-1),[0,2,1]))

