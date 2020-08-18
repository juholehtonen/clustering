# encoding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import (DictVectorizer)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

import preprocess
from hetero_feature_union import ItemSelector

data = preprocess.get_data()


vec = DictVectorizer()
data_vectorized = vec.fit_transform(data)
#hasher = FeatureHasher()
#data_vectorized = hasher.fit_transform(data)

class DisciplineExtractor(BaseEstimator, TransformerMixin):
    """Extract the discipline from data"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        transformed = []
        for d in data:
            transformed.append(d['discipline'])
        return transformed

n_cl = 4
pl = Pipeline([
    # ('selector', ItemSelector(key='discipline')),
    ('discipline', DisciplineExtractor()),
    ('vectorizer', CountVectorizer()),
    ('svd', TruncatedSVD(n_components=2))
    # ('kmeans', KMeans(init='k-means++', n_clusters=n_cl, n_init=10))
    ])

reduced_data  = pl.fit_transform(data)
# Get timings for different clusterings: full dimensions, dimension reduced

# Visualize the results
kmeans = KMeans(init='k-means++', n_clusters=n_cl, n_init=10)
kmeans.fit(reduced_data)
h = .02

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
plt.title(u'Suomalaisten tieteellisten julkaisujen ryhmittelyanalyysi K-means-\n'
          u'menetelmällä (baseline)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


fu = FeatureUnion(
    transformer_list=[

        # Pipeline for pulling features from publication's discipline key
        ('discipline', Pipeline([
            ('selector', ItemSelector(key='discipline')),
            ('vectorizer', CountVectorizer())

        ]))
    ]
)