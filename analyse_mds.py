# coding=UTF8
##################################################################
# Step 2: Analyze data with MDS
#
# Usage: python analyse_mds.py <label>
##################################################################
from time import time
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn import manifold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from utils import GeneralExtractor

# Read the label for a run.
label = sys.argv[1]

# Load data from the previous step
with open('../data/{0}-preprocessed.txt'.format(label), 'r') as handle:
  data = cPickle.load(handle)
# data = np.loadtxt('data/{0}-preprocessed.txt'.format(label))


# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 'x',
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig('../img/{0}-plot.png'.format(label), bbox_inches='tight')
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# MDS  embedding of the dataset
print("Computing MDS embedding")
# clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
pl_mds = Pipeline([('takeall', GeneralExtractor()),
                   ('vectorizer', CountVectorizer()),
                   ('densify', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                   ('mds', manifold.MDS(n_components=2, n_init=1, max_iter=100))
                   ])
X_mds = pl_mds.fit_transform(data)
# X_mds = clf.fit_transform(X)
#print("Done. Stress: %f" % pl_mds['mds'].stress_)
plot_embedding(X_mds,
               "MDS embedding of the data (time %.2fs)" %
               (time() - t0))
#----------------------------------------------------------------------
