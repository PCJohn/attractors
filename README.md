# attractors
A set of simulations on attractor models

**Reproducing Fast Weights results**

      cd repro_fw
      python retrieval.py gen_data # cache datasets in ./data
      python retrieval.py lstm_vs_fw 8 # run experiment with sequence of length 8

This will reproduce the results in section 4.1 of the paper with 30 hidden units and learned 64 dimensional embeddings (you can change parameters [here](https://github.com/PCJohn/attractors/blob/e1435ea0bd83a30f41cb22d8cb27992cfb83fd39/repro_fw/retrieval.py#L13)). 

<img src="repro_fw/outputs/seqlen-8.png" width="650" height="400" />
