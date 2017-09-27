# coding=UTF8
##################################################################
# Step 2: Analyze data with k-means
#
# Usage: python analyse_mds.py <label>
##################################################################
from time import time
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from utils import GeneralExtractor

# Read the label for a run.
# label = sys.argv[1]
label = 'baseline'

# Load data from the previous step
with open('../data/{0}-preprocessed.txt'.format(label), 'r') as handle:
  data = cPickle.load(handle)

n_cl = 6
pl = Pipeline([
    # ('discipline', DisciplineExtractor()),
    ('takeall', GeneralExtractor()),
    ('vectorizer', CountVectorizer()),
    ('svd', TruncatedSVD(n_components=2))   # comment out to check data
    ])
t0 = time()
reduced_data = pl.fit_transform(data)
# reduced_data[:,:10].toarray() # Test sanity of pipeline stages by commenting out some.
# Get timings for different clusterings: full dimensions, dimension reduced
t_preprocessing = time() - t0

#kmeans = KMeans(init='k-means++', n_clusters=n_cl, n_init=10)
kmeans = KMeans(init='random', n_clusters=n_cl, n_init=10)
kmeans.fit(reduced_data)
t_end = time() - t0; t1 = time()
print('training time:  %.2fs,    n_samples: %i' % (t_end, len(reduced_data)))

## Visualize the results
# flattend_data = TruncatedSVD(n_components=2).fit_transform(reduced_data)
h = .2
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in grid. Use last trained model.
t2 = time()
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
print('predict time: %.2fs' % (time() - t2))

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.title(u'Suomalaisten tieteellisten julkaisujen ryhmittelyanalyysi K-means-\n'
          u'menetelmällä (baseline)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
plt.savefig('../img/{0}-plot.png'.format(label), bbox_inches='tight')
print('plot time: %.2fs' % (time() - t1))