import numpy as np
from matplotlib import pyplot as plt

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

##### *** #####


##### Memory model #####

class Hopfield():
    def __init__(self,x,transitions=None):
        dim = x.shape[1]
        self.J = np.matmul(x.T,x) / float(dim)
        
        # Add cross-terms so the model hops around on points as discrete states
        if transitions is not None:
            lmbda = 3.0
            for i in range(transitions.shape[0]):
                for j in range(transitions[i].shape[0]):
                    if i != j:
                        self.J += lmbda * transitions[i,j] * np.outer(x[i][:,np.newaxis],x[j][:,np.newaxis]) / float(dim)
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




if __name__ == '__main__':
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
    seq = mem.lookup(x[0],n_step=seq_len) # prompt with the first symbol and generate a sequence
    seq_states = [get_state(x,s) for s in seq]
    print('Retrieved sequence:',seq_states)

