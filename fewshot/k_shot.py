import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from matplotlib import pyplot as plt

import tfutils as tfu

class Conv():
    def __init__(self, K, N, episode_len, S, S_lab, B, B_lab, mem):
        # K-shot vars -- see 2.2 of "Matching Networks"
        self.K = K              # num samples per class
        self.N = N              # num classes
        self.S = S              # support set
        self.S_lab = S_lab      # labels for support set
        self.B = B              # batch with novel samples
        self.B_lab = B_lab      # labels for batch
        self.bz = episode_len
        self.mem = mem
        self.embed_dim = 64
        self.lr = 1e-4
        self.n_iter = 100000
        self.num_meta_test_sets = 2000

        # Variables for memory module
        self.fast_lr = 1./(self.K*self.N)
        self.decay = 1. # no decay -- uniform weight over support set
        self.inner_loop = 1
        self.memmat = tf.zeros([self.embed_dim,self.embed_dim],dtype=tf.float32)
        self.ln_gain = tf.Variable(tf.ones([self.embed_dim]),trainable=True,dtype=tf.float32,name='ln_gain')
        self.ln_bias = tf.Variable(tf.zeros([self.embed_dim]),trainable=True,dtype=tf.float32,name='ln_bias')

        self.build_graph()
 
    # Insert to mem with the storkey update rule
    def insert(self,x):
        hebb = self.memmat + tfu.outer(x,x) / self.embed_dim
        h = tf.matmul(hebb,tf.expand_dims(x,-1))
        pre = tfu.outer(tf.squeeze(h,-1),x)
        post = tf.transpose(pre,perm=[0,2,1])
        self.memmat = self.decay * self.memmat + self.fast_lr * (hebb - pre - post) / self.embed_dim

    # Run attractor
    def recall(self,x):
        x_0 = tf.expand_dims(x,-1)
        x_s = x_0
        for s in range(self.inner_loop):
            step = x_0 + tf.matmul(self.memmat,x_s)
            x_s = tfu.layer_norm(step, self.ln_gain, self.ln_bias)
        x_s = tf.squeeze(x_s,-1)
        return x_s
    
    def recall_batch(self,bx):
        out = []
        for i in range(self.bz*self.N):
            out.append(self.recall(tf.expand_dims(bx[i],0)))
        out = tf.stack(out)
        return tf.squeeze(out,1)

    # Apply conv layers to get embeddings
    def conv_features(self,conv_w,x):
        embed = x
        for cw in conv_w:
            embed = tf.nn.conv2d(embed,cw[0],strides=[1,1,1,1],padding='SAME')
            embed = tf.nn.bias_add(embed,cw[1])
            embed = tf.nn.max_pool(embed,[1,3,3,1],[1,2,2,1],'SAME')
            embed = tf.nn.relu(embed)
        embed = tf.contrib.layers.flatten(embed)
        return embed

    def build_graph(self):
        # Conv weights
        conv_w = [tfu.conv_wts(3,3,1,64),
                  tfu.conv_wts(3,3,64,64),
                  tfu.conv_wts(3,3,64,64),
                  tfu.conv_wts(3,3,64,64),
                  tfu.conv_wts(3,3,64,64)]
        
        S_embed = self.conv_features(conv_w,self.S) # Embed support set
        # Insert support set in memory
        if self.mem:
            for s in range(self.K*self.N):
                self.insert(tf.expand_dims(S_embed[s],0)) 
        
        B_embed = self.conv_features(conv_w,self.B) # Embed batch
        # Recall from memory and modify embeddings
        if self.mem:
            B_embed = self.recall_batch(B_embed)

        # Similarity
        att_logits = tf.matmul(B_embed, tf.transpose(S_embed))
        self.att = tf.nn.softmax(att_logits)

        # Predict labels for batch
        self.pred_conf = tf.nn.softmax(tf.matmul(self.att,self.S_lab))
        self.pred = tf.argmax(self.pred_conf,axis=1,name='pred_op')
        self.loss = tf.losses.mean_squared_error(labels=self.B_lab,predictions=self.pred_conf)

        self.opt_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    def train(self,sess):
        for itr in range(self.n_iter):
            blab,py,loss,_ = sess.run([self.B_lab,self.pred,self.loss,self.opt_op])
            if itr%500 == 0:
                acc = np.mean(py == blab.argmax(axis=1))
                print('Itr '+str(itr)+': '+str(loss)+' '+str(acc))


    def test(self, sess):
        meta_acc = []
        for _ in range(self.num_meta_test_sets):
            vblab, vpy = sess.run([self.B_lab, model.pred])
            acc = np.mean(vpy == vblab.argmax(axis=1))
            meta_acc.append(acc)
        print('Test acc: Mean -',np.mean(meta_acc),'Std -',np.std(meta_acc))

def parse_args():
    parser = argparse.ArgumentParser(description='N-way, K-shot learning task')
    parser.add_argument('--dataset', default='omniglot', help='omniglot or miniimagenet')
    parser.add_argument('--N', type=int, default=5, help='Num classes')
    parser.add_argument('--K', type=int, default=1, help='Num samples per class')
    parser.add_argument('--ep_len', type=int, default=2, help='Num samples per class in episode')
    parser.add_argument('--use_mem', action='store_true', help='Use if we want to use a memory module')
    parser.add_argument('--output_dir', default='outputs', help='Folder to save outputs')
    parser.add_argument('--task', required=True, help='train or test')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    K = args.K
    N = args.N
    episode_len = args.ep_len
    mem = args.use_mem
    dataset = args.dataset
    output_dir = args.output_dir
    model_path = os.path.join(output_dir,'conv_mem-'+str(mem))
    test_set = (args.task=='test')
    
    if not os.path.exists(output_dir):
        os.system('mkdir '+output_dir)
    
    with tf.Session() as sess:
        data_generator = DataGenerator(K+episode_len,
                                       num_classes=N,
                                       datasource=dataset,
                                       test_set=test_set)
        S,S_lab,B,B_lab = data_generator.make_data_tensor(K,N,episode_len,test_set=test_set)
        
        # https://stackoverflow.com/questions/46880589/reading-image-files-into-tensorflow-with-tf-wholefilereader
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(coord=coord)

        model = Conv(K,N,
                     episode_len,
                     S,S_lab,B,B_lab,
                     mem)
        sess.run(tf.global_variables_initializer())

        # Train
        if not (test_set):
            model.train(sess)
            tf.train.Saver().save(sess, model_path)
        
        # Test
        else:
            tf.train.Saver().restore(sess, model_path)
            model.test(sess)
   
    sess.close()


