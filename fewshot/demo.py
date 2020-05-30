import os
import sys
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from matplotlib import pyplot as plt

from k_shot import Conv
import tfutils as tfu


if __name__ == '__main__':
    K = 1
    N = 5
    episode_len = 2
    mem = True
   
    output_dir = 'outputs'
    model_path = os.path.join(output_dir,'conv_mem-'+str(mem))
    
    ds = np.load('mnist_1_2.npy')
    ds = list(map(np.float32,ds))
      
    with tf.Session() as sess:
        S,S_lab,B,B_lab = list(map(tf.convert_to_tensor, ds))
        model = Conv(K,N,
                     episode_len,
                     S,S_lab,B,B_lab,
                     mem)
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(sess, model_path)
        blab,plab = sess.run([B_lab,model.pred])
    sess.close()

    # Display support set
    for i in range(N):
        ax = plt.subplot(100+N*10+i+1)
        plt.imshow(ds[0][i][:,:,0],cmap='gray')
        ax.set_title(str(ds[1][i].argmax()))
        ax.axis('off')
    plt.suptitle('Support Set')
    plt.show()

    # Display test samples and predictions
    for i in range(episode_len):
        for j in range(N):
            ax = plt.subplot2grid((episode_len,N), (i,j))
            ind = N*i+j
            print(ind)
            plt.imshow(ds[2][ind][:,:,0],cmap='gray')
            ax.set_title(str(ds[3][ind].argmax()))
            ax.axis('off')
    plt.suptitle('Predictions')
    plt.show()



