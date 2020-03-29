from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import rnn_mem

# Add noise to the query: last character in the input sequence
def add_query_noise(x,level):
    x_ = x.copy()
    x_[:,-1,:] += np.random.normal(0,level,size=x[:,-1,:].shape)
    return x_


if __name__ == '__main__':
    ds = np.load('data/keyval_8.npz')
    tx,ty = ds['testx'],ds['testy']
   
    # Add noise to queries
    noise_levels = np.arange(0,1.0,0.05)
    noisy_x = [add_query_noise(tx,level) for level in noise_levels]
    models = ['outputs/LSTM_seqlen-8','outputs/FW_seqlen-8']
    inner_loops = [0,1]
    names = ['LSTM','FW']
    
    model_acc = {m : [] for m in models}
    for model_file,inner_loop in zip(models,inner_loops):
        with tf.Session() as sess:
            if 'FW' in model_file:
                model = rnn_mem.FastWeights(tx.shape[1],tx.shape[2],learn_embed=False,shared_ln=False,inner_loop=inner_loop)
            elif 'LSTM' in model_file:
                model = rnn_mem.LSTM(tx.shape[1],tx.shape[2],learn_embed=False)
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess,model_file)
            for nx,noise_level in zip(noisy_x,noise_levels):
                py = sess.run(model.pred, feed_dict={model.x:nx})
                acc = np.mean(py==ty)
                model_acc[model_file].append(acc)
        tf.reset_default_graph()
        sess.close()
    
    plt.ylim((0,1))
    for m,name in zip(models,names):
        plt.plot(noise_levels,model_acc[m],marker='o',label=name)
    plt.legend()
    plt.ylabel('Test Acc.')
    plt.xlabel('Noise Std.')
    plt.show()



