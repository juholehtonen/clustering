# coding=UTF8
##################################################################
# Step 2: Cluster data with hierarchical Ward's method.
#
# Usage: python analyse_hierarchical.py <size>
##################################################################
from __future__ import print_function

import logging
from nltk.corpus import stopwords
import numpy as np
from optparse import OptionParser
import pandas as pd
import pickle
import random
import scipy.sparse
import sys
from time import time

from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (Normalizer,
                                   FunctionTransformer)

from lemmatizer import NLTKPreprocessor
from utils import (GeneralExtractor,
                   plot_silhouette,
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
              help="Output directory for results")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")


# print(__doc__)
# op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
# argv = [] if is_interactive() else sys.argv[1:]
# (opts, args) = op.parse_args(argv)
(opts, args) = op.parse_args()
# if len(args) > 0:
#     op.error("this script takes no arguments.")
#     sys.exit(1)
print('Options: {ops}'.format(ops=opts))

# Define log file name and start log. NOTE: 'w' overwrites every time!!
#'--size {0} --n-clusters {1} --lsa {2} --n-features {3} --fields {4}'\
#              .format(size, k, n_comp, n_feat, analysis_fields)
results_prefix = '{0}-{1}-'.format(opts.size, opts.n_clusters)
if opts.n_components:
    results_prefix += str(opts.n_components) + '-'
results_file = opts.out + results_prefix + 'hierarchical-results.txt'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M",
                    filename=results_file,
                    filemode='w')
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
ward = AgglomerativeClustering(n_clusters=opts.n_clusters,
                               linkage='ward', affinity='euclidean', connectivity=None,
                               # memory=None, compute_full_tree=False
                               )

logging.info("Compute unstructured hierarchical clustering with %s" % ward)
t0 = time()
ward.fit(X)
logging.info("  Done in %0.3fs" % (time() - t0))

# Calculate metrics
t0 = time()
model_dir = '../models/'
truth_file = model_dir + 'groundtruth_labels_final.csv'
labels = pd.read_csv(truth_file, index_col=0).values[:,0].tolist()

silhouette_list = []

logging.info("  Silhouette Coefficient: %0.3f"
             % metrics.silhouette_score(X, ward.labels_, sample_size=1000))
X_to_CH = X if opts.n_components else X.toarray()
logging.info("  Calinski-Harabasz Index: %0.3f"
             % metrics.calinski_harabasz_score(X, ward.labels_))
logging.info("  Adjusted Rand-Index: %0.3f"
             % metrics.adjusted_rand_score(labels, ward.labels_))
logging.info("  Metrics calculated in %fs" % (time() - t0))


# Top terms per cluster
t0 = time()
logging.info("Top terms per cluster:")
tficf = pd.DataFrame(X)
tficf['label'] = ward.labels_
cluster_term_means = tficf.groupby('label').mean()
if opts.n_components:
    original_space_term_means = svd.inverse_transform(cluster_term_means)
    sorted_ctms = np.argsort(original_space_term_means)
else:
    sorted_ctms = np.array(np.argsort(cluster_term_means))
for i in range(opts.n_clusters):
    top_for_i = sorted_ctms[i, ::-1][:15]
    term_str = ' '.join([terms[j] for j in top_for_i])
    logging.info("  Cluster {0}: {1}".format(i, term_str))
logging.info("  Done in %fs" % (time() - t0))


# Sample of publications per cluster
logging.info('Sample of publications per cluster:')
t0 = time()
sample_max = 5
with open(opts.source, 'rb') as handle:
    data = pickle.load(handle)
for k in range(opts.n_clusters):
    cluster_k_pubs = [d for (d, l) in zip(data, ward.labels_) if l == k]
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

# Plot silhouette
plot_silh = False
if plot_silh:
    t0 = time()
    plot_silhouette(X, ward.labels_, opts.n_clusters, 'Hierarchical')
    logging.info("Silhouette plot in %fs" % (time() - t0))

logging.info("Total running time: %fs" % (time() - t_total))