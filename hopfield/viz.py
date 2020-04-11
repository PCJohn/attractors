import cv2
import numpy as np
from matplotlib import pyplot as plt

def save_vid(frames,out_file):
    out_size = (640,480)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(out_file,fourcc,1,out_size)
    for f in frames[:10]:
        f = cv2.resize(f,out_size)
        out.write(f)
    out.release()

def disp_states(attractors,sequence,save_file):
    embed_dim = attractors.shape[1]
    
    # Fixed, random projection down to 2D
    proj = np.random.normal(0,1,size=(embed_dim,2))

    # Project attractors
    att = np.matmul(attractors,proj)
    attx,atty = att[:,0],att[:,1]
    
    # Plot sequence
    frames = []
    for s in sequence:
        fig = plt.figure()
        # Display attractors
        plt.scatter(attx,atty,marker='x',color='r',s=150)
        plt.ylim((-20,20))
        plt.xlim((-20,20))
        # Display state
        state = np.matmul(np.array(s)[np.newaxis,:],proj)
        sx,sy = state[:,0],state[:,1]
        plt.scatter([sx],[sy],marker='o',color='b',s=75)
        #plt.show()
        fig.canvas.draw()
        f = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        f = f.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(f)
        plt.clf()
        plt.close()
    # Save video
    save_vid(frames,save_file)

