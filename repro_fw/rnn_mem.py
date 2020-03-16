from __future__ import print_function
import numpy as np
import tensorflow as tf

import tfutils as tfu

class LSTM():
    def __init__(self,seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.x = tf.placeholder(tf.float32,shape=(None,seq_len,dim),name='input_ph')
        self.y = tf.placeholder(tf.int32,shape=(None,dim),name='label_ph')
        self.hid_size = 64
        self.lr = 1e-3
        self.bz = 64
        self.niter = 75000
        self.build_graph()

    def build_graph(self):
        outputs, (c,h) = tf.nn.dynamic_rnn(cell=tf.contrib.rnn.LSTMCell(self.hid_size,activation=tfu.relu),
                                inputs=self.x,
                                dtype=tf.float32)
        out = tfu.dense(c,100,act=tfu.relu)
        logits = tfu.dense(out,self.dim)
        self.pred =  tfu.softmax(logits)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y,logits=logits)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.opt_op = opt.minimize(self.loss)


class FastWeights():
    def __init__(self,seq_len,dim):
        self.seq_len = seq_len
        self.dim = dim
        self.x = tf.placeholder(tf.float32,shape=(None,seq_len,dim),name='input_ph')
        self.y = tf.placeholder(tf.int32,shape=(None,dim),name='label_ph')
        self.hid_size = 64
        self.memmat = tf.eye(self.hid_size) * 0.05
        self.hid = tf.zeros(shape=(tf.shape(self.x)[0],self.hid_size),dtype=tf.float32)
        self.inner_loop = 1
        self.lr = 1e-3
        self.fast_lr = 0.5
        self.decay = 0.9
        self.bz = 64
        self.niter = 75000
        self.build_graph()
                
    def build_graph(self):
        for t in range(self.seq_len):
            x_t = self.x[:,t,:]
            z = tfu.dense(self.hid,self.hid_size,act=None) + tfu.dense(x_t,self.hid_size,act=None)
            h_0 = tfu.relu(z)

            # insert in memory
            self.memmat = self.decay * self.memmat + self.fast_lr * tfu.outer(self.hid,self.hid)
            
            # run attractor
            h_s = tf.expand_dims(h_0,-1)
            z_ = tf.expand_dims(z,-1)
            for _ in range(self.inner_loop):
                h_s = tfu.layer_norm(z_ + tf.matmul(self.memmat,h_s))
                h_s = tfu.relu(h_s)
            h_s = tf.squeeze(h_s,-1)

            # update hidden state
            self.hid = h_s
        
        out = tfu.dense(self.hid,100,act=tfu.relu)
        logits = tfu.dense(out,self.dim)
        self.pred =  tfu.softmax(logits)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y,logits=logits)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.opt_op = opt.minimize(self.loss)


def train(sess,model,x,y,vx,vy):
    sess.run(tf.global_variables_initializer())
    val_t,loss_t = [],[]
    for itr in range(model.niter):
        bi = np.random.randint(0,x.shape[0],model.bz)
        bx,by = x[bi],y[bi]
        
        feed_dict = {model.x:bx, model.y:by}
        sess.run(model.opt_op,feed_dict=feed_dict)
        if itr % 500 == 0:
            py,vloss = sess.run([model.pred,model.loss],feed_dict={model.x:vx,model.y:vy})
            vacc = np.mean(py.argmax(1)==vy.argmax(1))
            print('\tIteration:',itr,'\t',vacc,vloss)
            val_t.append(vacc)
            loss_t.append(vloss)
    return val_t,loss_t



