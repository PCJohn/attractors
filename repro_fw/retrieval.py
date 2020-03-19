from __future__ import print_function
import os
import sys
import string
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import rnn_mem

##### Params #####
n_train = 100000
n_val = 10000
n_test = 20000
seq_lens = range(2,22,2)
data_dir = './data'
output_dir = './outputs'
delim = '?'
learn_embed = True
##### *** #####

def ds_file(seq_len):
    return 'keyval_'+str(seq_len)+'.npz'

non_delim_chars = list(string.ascii_lowercase)
charlist = non_delim_chars+[delim]
intlist = range(10)
dim = len(charlist) + len(intlist)

embed_mat = np.eye(dim)
embed = {c : embed_mat[i] for i,c in enumerate(charlist)}
embed.update({str(d) : embed_mat[len(charlist)+i] for i,d in enumerate(intlist)})
char_ids = {c : i for i,c in enumerate(charlist)}
char_ids.update({str(d) : len(charlist)+i for i,d in enumerate(intlist)})

def gen_seq(K):
    np.random.shuffle(non_delim_chars)
    np.random.shuffle(intlist)
    seq = []
    for c,i in zip(non_delim_chars[:K//2],intlist[:K//2]):
        seq.append(c)
        seq.append(str(i))
    query = np.random.choice(range(0,K-1,2))
    key,value = seq[query],seq[query+1]
    return ''.join(seq)+delim+key,value

def gen_ds(seq_len):
    x,y = [],[]
    for _ in range(n_train+n_val+n_test):
        seq,v = gen_seq(seq_len)
        x.append(np.array([embed[c] for c in seq]))
        #y.append(embed[v])
        y.append(char_ids[v])
    x,y = np.array(x),np.array(y)
    train = [x[:n_train],y[:n_train]]
    val = [x[n_train:n_train+n_val],y[n_train:n_train+n_val]]
    test = [x[-n_test:],y[-n_test:]]
    return train,val,test

def plot_val_and_loss(val,loss,label):
    plt.subplot(121)
    plt.plot(loss,label=label)
    plt.ylim((0,2))
    plt.ylabel('Loss',labelpad=0)
    plt.xlabel('Iterations x 1000')
    plt.subplot(122)
    plt.plot(val,label=label)
    plt.ylim((0,1))
    plt.ylabel('Val. acc.',labelpad=0)
    plt.xlabel('Iterations x 1000')

if __name__ == '__main__':
    
    if sys.argv[1] == 'gen_data':
        for seq_len in seq_lens:
            print('Generating dataset with sequence length:',seq_len)
            train,val,test = gen_ds(seq_len)
            print('...Done. train, val, test shapes:')
            print(train[0].shape,train[1].shape,val[0].shape,val[1].shape,test[0].shape,test[1].shape)
            print('Saving...')
            if not os.path.exists(data_dir):
                os.system('mkdir '+data_dir)
            np.savez(os.path.join(data_dir,ds_file(seq_len)),
                trainx=train[0],trainy=train[1],valx=val[0],valy=val[1],testx=test[0],testy=test[1])
            print('...Done')

    elif sys.argv[1] == 'lstm_vs_fw':
        seq_len = int(sys.argv[2])
        ds = np.load(os.path.join(data_dir,ds_file(seq_len)))
        x,y,vx,vy = ds['trainx'],ds['trainy'],ds['valx'],ds['valy']
        in_len = x.shape[1]
        embed_dim = x.shape[2]
        rnn_models = ['LSTM','FW'][::-1]
        for modelname in rnn_models:
            with tf.Session() as sess:
                if modelname == 'LSTM':
                    model = rnn_mem.LSTM(in_len, embed_dim, learn_embed=learn_embed)
                elif modelname == 'FW':
                    model = rnn_mem.FastWeights(in_len, embed_dim, learn_embed=learn_embed)
                val,loss = rnn_mem.train(sess,model,x,y,vx,vy)
                plot_val_and_loss(val,loss,modelname)
                final_val = val[-1]
                tf.train.Saver().save(sess,os.path.join(output_dir,modelname+'_seqlen-'+str(seq_len)))
        plt.legend(bbox_to_anchor=(1.0,1.11),ncol=2,fontsize='small')

        if not os.path.exists(output_dir):
            os.system('mkdir '+output_dir)
        plt.savefig(os.path.join(output_dir,'seqlen-'+str(seq_len)+'.png'))


