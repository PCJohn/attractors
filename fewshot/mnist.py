import os
import cv2
import sys
import random
import numpy as np

path = '/home/prithvi/dsets/MNIST/trainingSet/'

def load(train_count=3000,val_count=1000,num_classes=10,one_hot=True):
    ds = []
    vds = []
    classes = os.listdir(path)
    random.shuffle(classes)
    classes = classes[:num_classes]
    unit = np.diag(np.ones(len(classes)))
    for n,c in enumerate(classes):
        n_path = os.path.join(path,c)
        if one_hot:
            lab = unit[int(n)]
        else:
            lab = int(n)
        flist = os.listdir(n_path)
        random.shuffle(flist)
        for s in flist[:train_count]:
            img = cv2.imread(os.path.join(n_path,s),0)
            img = np.float32(img)/255.
            img = img[:,:,np.newaxis]
            ds.append((img,lab))
        for s in flist[train_count:train_count+val_count]:
            img = cv2.imread(os.path.join(n_path,s),0)
            img = np.float32(img)/255.
            img = img[:,:,np.newaxis]
            vds.append((img,lab))
    random.shuffle(ds)
    random.shuffle(vds)
    x,y = map(np.array,zip(*ds))
    vx,vy = map(np.array,zip(*vds))
    return (x,y,vx,vy)

if __name__ == '__main__':
    train_count = 1
    val_count = 2
    num_classes = 5
    x,y,vx,vy = load(train_count=train_count,
                     val_count=val_count,
                     num_classes=num_classes,
                     one_hot=True)
    np.save('./mnist_'+str(train_count)+'_'+str(val_count),[x,y,vx,vy])



