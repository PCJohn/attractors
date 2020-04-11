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
    train_seq_lens = [4,8,12]
    test_seq_lens = range(2,22,2)
   
    #model_dir = 'outputs/hid50/' # models trained with hidden size 50
    model_dir = 'outputs/hid30/' # models trained with hidden size 30

    for train_seq_len in train_seq_lens:
        model_file = os.path.join(model_dir,'FW_seqlen-'+str(train_seq_len))
    
        seq_acc = []
    
        for seq_len in test_seq_lens:
        
            ds = np.load('data/keyval_'+str(seq_len)+'.npz')
            tx,ty = ds['testx'],ds['testy']

            with tf.Session() as sess:
                model = rnn_mem.FastWeights(tx.shape[1],tx.shape[2],learn_embed=False,shared_ln=True,inner_loop=1)
                sess.run(tf.global_variables_initializer())
                tf.train.Saver().restore(sess,model_file)
            
                py = sess.run(model.pred, feed_dict={model.x:tx})
                acc = np.mean(py==ty)
                seq_acc.append(acc)
        
            tf.reset_default_graph()
            sess.close()
    
        plt.ylim((0,1))
        plt.plot(test_seq_lens,seq_acc,marker='o',label='Train seq len: '+str(train_seq_len))
    plt.legend(loc='lower center')
    plt.ylabel('Acc.')
    plt.xlabel('Test seq len')
    plt.show()



