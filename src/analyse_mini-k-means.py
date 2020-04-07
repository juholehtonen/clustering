"""
=======================================
Clustering text documents using k-means
=======================================
From: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html

This is an example showing how the scikit-learn can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce dimensionality
and discover latent patterns in the data.

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function
import logging
from nltk.corpus import stopwords
import pickle
from optparse import OptionParser
import random
import scipy.sparse
import sys
from time import time

from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from lemmatizer import NLTKPreprocessor
from utils import (GeneralExtractor,
                   plot_silhouette,
                   identity_tokenizer,
                   LemmaTokenizer)


# parse commandline arguments
op = OptionParser()
op.add_option("--size",
              dest="size", type="int", default=400,
              help="Size of the preprocessed data to be used.")
op.add_option("--fields",
              dest="fields", type="string",
              help="Metadata fields to run analysis with.")
op.add_option("--n-clusters",
              dest="n_clusters", type="int", default=16,
              help="Number of clusters to be used.")
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--max-df", dest="max_df", type=float, default=0.1,
              help="TfidfVectorizer's max_df parameter")
op.add_option("--source",
              dest="source", type="string",
              help="Source file to process")
op.add_option("--interim",
              dest="interim", type="string",
              help="Interim folder")
op.add_option("--out",
              dest="out", type="string",
              help="Output dircetory for results")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

# print(__doc__)
# op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

# Define log file name and start log
results_filename = opts.out
for o in [opts.size, opts.n_clusters]:
    results_filename = results_filename + str(o) + '-'
if opts.n_components:
    results_filename = results_filename + str(opts.n_components) + '-'
results_filename += 'kmeans-results.txt'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M",
                    filename=results_filename)
logging.info('#' * 18 + ' Starting clustering  ' + '#' *18)

# #############################################################################
# Load data from the previous step
logging.info("Loading vectorized data")
t_total = time()
min_df = 2
X = scipy.sparse.load_npz(opts.interim + '{0}-{1}-{2}-{3}-vectorized.npz'.format(opts.size, min_df, opts.max_df, opts.n_features))
with open(opts.interim + '{0}-vectorizer_feature_names.pickle'.format(opts.size), 'rb') as handle:
    terms = pickle.load(handle)

# #############################################################################
# Dimensionality reduction
if opts.n_components:
    logging.info("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    logging.info("  Done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    logging.info("  Explained variance of the SVD step: {0}% with {1} components".format(
        int(explained_variance * 100), opts.n_components))


# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=opts.n_clusters, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=opts.n_clusters, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

logging.info("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
logging.info("  Done in %0.3fs" % (time() - t0))


#logging.info("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
#logging.info("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
#logging.info("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
#logging.info("Adjusted Rand-Index: %.3f"
#      % metrics.adjusted_rand_score(labels, km.labels_))
logging.info("  Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
X_to_CH = X if opts.n_components else X.toarray()
logging.info("  Calinski-Harabasz Index: %0.3f"
      % metrics.calinski_harabasz_score(X_to_CH, km.labels_))


logging.info("Top terms per cluster:")
if opts.n_components:
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
#        logging.info("original_space_centroids: \n{0}".format(original_space_centroids[:, ::600]))
#        logging.info("order_centroids: \n{0}".format(order_centroids[:, ::600]))
else:
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for k in range(opts.n_clusters):
    top_for_k = ' '.join([ terms[j] for j in order_centroids[k, :15] ])
    logging.info("  Cluster {0}: {1}".format(k, top_for_k))


logging.info('Sample of publications per cluster:')
t0 = time()
sample_max = 5
with open(opts.source, 'rb') as handle:
    data = pickle.load(handle)
for k in range(opts.n_clusters):
    cluster_k_pubs = [d for (d, l) in zip(data, km.labels_) if l == k]
    sample_size = sample_max if len(cluster_k_pubs) > sample_max else len(cluster_k_pubs) - 1
    pubs_sample = random.sample(cluster_k_pubs, sample_size)
    logging.info('  Cluster {0}:'.format(k))
    # FIXME: Tie publication's info columns to available fields
    for p in pubs_sample:
        pub_info = '          ' + p['title'][:80]\
                     + (80 - len(p['title'])) * ' ' + '|'
        pub_info += p['discipline'][:30] if p.get('discipline') else ''
        logging.info(pub_info)
logging.info("  Done in %fs" % (time() - t0))

#with open('../data/processed/{0}-results.txt'.format(opts.size),
# 'w') as handle:
#    handle.write(results)

plot_silhouette(X, km.labels_, opts.n_clusters, 'K-Means')
logging.info("Total running time: %fs" % (time() - t_total))