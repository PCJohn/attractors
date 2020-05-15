import sys
import numpy as np
from matplotlib import pyplot as plt

import viz

##### Utils #####

# Assign a random binary vector to each symbol in a sequence
def embed_nodes(A,dim):
    embeddings = np.random.random(size=(A.shape[0],dim))
    embeddings[embeddings<0.5] = -1
    embeddings[embeddings>0.5] = 1
    return embeddings
       
# Use nearest neighbours to identify the closest state
def get_state(embeddings,x):
    dist = [((x-e)**2).sum() for e in embeddings]
    return np.argmin(dist)

# Accuracy
def accuracy(pred,anchors):
    corr = 0
    for i,p in enumerate(pred):
        dist = [((p-x)**2).sum() for x in anchors]
        pred_anchor = np.argmin(dist)
        if pred_anchor == i:
            corr += 1
    return corr / float(anchors.shape[0])

# Add noise: flip some fraction of bits
def flip_noise(x,noise_level):
    noise = np.random.random(size=x.shape)
    x[noise<noise_level] *= -1
    return x

##### *** #####


##### Memory model #####

class Hopfield():
    def __init__(self,x,rule='hopfield',transitions=None):
        dim = x.shape[1]
        if rule == 'hopfield':
            self.J = np.matmul(x.T,x) / float(dim)
            # Add cross-terms so the model hops around on points as discrete states
            if not (transitions is None):
                lmbda = 3.0
                for i in range(transitions.shape[0]):
                    for j in range(transitions[i].shape[0]):
                        if i != j:
                            self.J += lmbda * transitions[i,j] * np.outer(x[i][:,np.newaxis],x[j][:,np.newaxis]) / float(dim)
            np.fill_diagonal(self.J,0)
        
        elif rule == 'storkey':
            self.J = np.zeros((dim,dim))
            for x_ in x:
                x_ = x_[np.newaxis,:]
                h = np.matmul(x_,self.J+np.matmul(x_.T,x_)/float(dim))
                pre = np.matmul(x_.T,h)
                post = pre.T
                self.J += (np.matmul(x_.T,x_) - pre - post) / float(dim)
            np.fill_diagonal(self.J,0) 
              
    def lookup(self, x, n_step=5, return_seq=False):
        out = x
        if return_seq:
            seq = []
        for _ in range(n_step):
            out = np.matmul(out, self.J)
            out[out<0] = -1
            out[out>0] = 1
            if return_seq:
                seq.append(out)
        if return_seq:
            return seq
        else:
            return out

##### *** #####


##### Comparing update rules #####
def compare_update_rules(dim=100,prob_one=0.5):
    num_mem = range(5,dim,5)
    for update_rule in ['hopfield','storkey']:
        n_mem_vs_acc = []
        for n_mem in num_mem:
            rule_acc = []
            for _ in range(20):
                # Generate random vectors with -1, +1 to save in memory
                rand_x = np.random.random(size=(n_mem,dim))
                rand_x[rand_x<(1-prob_one)] = -1
                rand_x[rand_x>prob_one] = 1
                mem = Hopfield(rand_x,rule=update_rule)
                pred = mem.lookup(rand_x,n_step=30) # try to recover mem by lookup
                rule_acc.append(accuracy(pred,rand_x))
            n_mem_vs_acc.append(np.mean(rule_acc))
        plt.plot(num_mem,n_mem_vs_acc,marker='o',label=update_rule)
    # Plot recall accuracy num mem increases
    plt.xlabel('Number of memories')
    plt.ylabel('Recall accuracy')
    plt.ylim((0,1.05))
    plt.legend()
    plt.show()

##### *** #####




if __name__ == '__main__':
    # Standard Hopfield for memory lookup
    if sys.argv[1] == 'lookup':
        num_mem = 5
        embed_dim = 100
        # Generate random zero-centred vectors to save in memory
        mem_x = np.random.random(size=(num_mem,embed_dim))
        mem_x[mem_x<0.5] = -1
        mem_x[mem_x>0.5] = 1
        mem = Hopfield(mem_x)
        # Add noise to the attractors -- use these as queries to lookup in memory
        queries = np.array([flip_noise(x,0.30) for x in mem_x])
        lookup_seq = mem.lookup(queries,n_step=10,return_seq=True)
        r = np.random.randint(queries.shape[0]) # pick a random query to plot
        lookup_seq = [l[r] for l in lookup_seq]
        # Visualize transitions for lookup
        viz.disp_states(mem_x,lookup_seq,'./lookup.mp4')

    # Hopfield net with discrete state jumps 
    elif sys.argv[1] == 'sequence':
        state_transitions = np.array([
                            [0,1,0,0,0],
                            [0,0,1,0,0],
                            [0,0,0,1,0],
                            [0,0,0,0,1],
                            [1,0,0,0,0]])
        embed_dim = 100
        x = embed_nodes(state_transitions,embed_dim) # assign random binary vectors to each symbol
        mem = Hopfield(x,transitions=state_transitions)
    
        # Generate a sequence obeying the transitions matrix above
        seq_len = 40
        seq = mem.lookup(x[0],n_step=seq_len,return_seq=True) # prompt with the first symbol and generate a sequence
        seq_states = [get_state(x,s) for s in seq]
        print('Retrieved sequence:',seq_states)

        # Visualize state transitions
        viz.disp_states(x,seq,'./sequence.mp4')

    # Compare memory capacity with different update rules
    elif sys.argv[1] == 'update_rule':
        compare_update_rules(dim=100,prob_one=0.4)


