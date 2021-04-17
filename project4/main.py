import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
K = 4
seed = 4
mixture, post = common.init(X,K,seed)
print(mixture)
mixture, post, cost = kmeans.run(X,mixture,post)
print(mixture)
print(cost)
common.plot(X,mixture,post,"K = "+str(K)+"\nseed = "+str(seed)+"\ncost = "+ str(cost))
# TODO: Your code here
