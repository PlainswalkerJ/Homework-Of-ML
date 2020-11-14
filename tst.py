import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Trees
import treePlotter

data,labels=Trees.createDataSet()
tree=treePlotter.retrieveTree(0)
print(data,labels,tree)
print(Trees.classify(tree,labels,[1,1]))




