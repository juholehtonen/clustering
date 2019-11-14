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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import (DictVectorizer,
                                        FeatureHasher)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Read the label for a run.
# label = sys.argv[1]
label = 'baseline'
size = 120

# Load data from the previous step
with open('data/{0}-preprocessed.txt'.format(size), 'r') as handle:
  data = cPickle.load(handle)
# data = np.loadtxt('data/{0}-preprocessed.txt'.format(label))

class DisciplineExtractor(BaseEstimator, TransformerMixin):
    """Extract the discipline from data"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        transformed = []
        for d in data:
            transformed.append(d['discipline'])
        return transformed


class GeneralExtractor(BaseEstimator, TransformerMixin):
    """Extract and merge all fields from a sample to a string"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        transformed = []
        for d in data:
            transformed.append(' '.join(d.values()))
        return transformed


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
z_min, z_max = reduced_data[:, 2].min() - 1, reduced_data[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))

# Obtain labels for each point in grid. Use last trained model.
t2 = time()
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
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
print('plot time: %.2fs' % (time() - t1))