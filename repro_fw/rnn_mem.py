from __future__ import print_function
import numpy as np
import tensorflow as tf

import tfutils as tfu

# Stem to learn embeddings -- use same for LSTM and FW
def embed(inp,embed_size):
    batch_size,seq_len,dim = inp.shape.as_list()
    w1 = tf.Variable(tfu.xinit([dim,50]),trainable=True,dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([50]),trainable=True,dtype=tf.float32)
    w2 = tf.Variable(tfu.xinit([50,embed_size]),trainable=True,dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([embed_size]),trainable=True,dtype=tf.float32)
    outp = []
    for t in range(seq_len):
        xt = inp[:,t,:]
        l1 = tfu.relu(tf.matmul(xt, w1) + b1)
        l2 = tfu.relu(tf.matmul(l1, w2) + b2)
        outp.append(l2)
    embeddings = tf.stack(outp,axis=1)
    return embeddings

class LSTM():
    def __init__(self,seq_len,dim,learn_embed=False):
        self.seq_len = seq_len
        self.dim = dim
        self.learn_embed = learn_embed
        self.x = tf.placeholder(tf.float32,shape=(None,seq_len,dim),name='input_ph')
        self.y = tf.placeholder(tf.int32,shape=(None,),name='label_ph')
        self.embed_size = 64
        self.hid_size = 30
        self.lr = 1e-3
        self.bz = 64
        self.niter = 50000
        self.build_graph()

    def build_graph(self):
        if self.learn_embed:
            in_seq = embed(self.x,self.embed_size)
        else:
            in_seq = self.x
        outputs, (c,h) = tf.nn.dynamic_rnn(cell=tf.contrib.rnn.LSTMCell(self.hid_size,activation=tfu.relu),
                                inputs=in_seq,
                                dtype=tf.float32)
        out = tfu.dense(c,100,act=tfu.relu)
        logits = tfu.dense(out,self.dim)
        self.pred = tf.argmax(tfu.softmax(logits),axis=1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y,logits=logits)
        self.opt_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


class FastWeights():
    def __init__(self,seq_len,dim,learn_embed=False):
        self.seq_len = seq_len
        self.dim = dim
        self.learn_embed = learn_embed
        self.x = tf.placeholder(tf.float32,shape=(None,seq_len,dim),name='input_ph')
        self.y = tf.placeholder(tf.int32,shape=(None,),name='label_ph')
        self.embed_size = 64
        self.hid_size = 30
        self.memmat = tf.zeros([self.hid_size,self.hid_size],dtype=tf.float32)
        self.hid = tf.zeros(shape=(tf.shape(self.x)[0],self.hid_size),dtype=tf.float32)
        self.inner_loop = 1
        self.lr = 1e-4
        self.fast_lr = 0.5
        self.decay = 0.9
        self.bz = 64
        self.niter = 50000
        self.build_graph()
                
    def build_graph(self):
        if self.learn_embed:
            in_seq = embed(self.x,self.embed_size)
            C = tf.Variable(tfu.xinit([self.embed_size,self.hid_size]),trainable=True,dtype=tf.float32)
        else:
            in_seq = self.x
            C = tf.Variable(tfu.xinit([self.dim,self.hid_size]),trainable=True,dtype=tf.float32)

        W = tf.Variable(0.05*tf.eye(self.hid_size),trainable=True,dtype=tf.float32)
        
        for t in range(self.seq_len):
            x_t = in_seq[:,t,:]
            
            z = tf.matmul(self.hid,W) + tf.matmul(x_t,C)
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
        self.pred =  tf.argmax(tfu.softmax(logits),axis=1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y,logits=logits)
        self.opt_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


def train(sess,model,x,y,vx,vy):
    sess.run(tf.global_variables_initializer())
    val_t,loss_t = [],[]
    for itr in range(model.niter):
        bi = np.random.randint(0,x.shape[0],model.bz)
        bx,by = x[bi],y[bi]
        
        feed_dict = {model.x:bx, model.y:by}
        sess.run(model.opt_op,feed_dict=feed_dict)
        if itr % 1000 == 0:
            py,vloss = sess.run([model.pred,model.loss],feed_dict={model.x:vx,model.y:vy})
            vacc = np.mean(py==vy)
            print('\tIteration:',itr,'\t',vacc,vloss)
            val_t.append(vacc)
            loss_t.append(vloss)
    return val_t,loss_t



