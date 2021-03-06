# coding=UTF8
##################################################################
# Step 2: Cluster data with hierarchical Ward's method.
#
# Usage: python analyse_hierarchical.py <size>
##################################################################
from __future__ import print_function

import logging
from matplotlib import pyplot as plt
import numpy as np
from optparse import OptionParser
import pandas as pd
import pickle
import random
from s_dbw import S_Dbw
import sys
from time import time

from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

import plot_dendrogram as pden
from utils import plot_silhouette


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
              help="Number of clusters to be used or None for full tree.")
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
op.add_option("--min-df",
              dest="min_df", type=int, default=2,
              help="TfidfVectorizer's min_df parameter")
op.add_option("--source",
              dest="source", type="string",
              help="Source file to process")
op.add_option("--preproc-file",
              dest="preproc_file", type="string",
              help="Pre-processed file to for example titles")
op.add_option("--svd-file",
              dest="svd_file", type="string",
              help="SVD of the data")
op.add_option("--interim",
              dest="interim", type="string",
              help="Interim folder")
op.add_option("--out",
              dest="out", type="string",
              help="Output directory for results")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")
op.add_option("--baseline",
              action="store_true", dest="baseline", default=False,
              help="Define run as baseline run. We have ground truth (for ARI).")


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
logging.info("Loading vectorized & reduced data")
t_total = time()
vect_reduced_file = opts.source
term_file = opts.interim + '{0}-{1}-{2}-{3}-vectorizer_feature_names.pickle'.format(opts.size, opts.min_df, opts.max_df, opts.n_features)
svd_file = opts.svd_file
logging.info("   vectorized data file: {0}".format(vect_reduced_file))
logging.info("   term file: {0}".format(term_file))
logging.info("   SVD file: {0}".format(svd_file))
with open(vect_reduced_file, 'rb') as handle:
    X = pickle.load(handle)
with open(term_file, 'rb') as handle:
    terms = pickle.load(handle)
with open(svd_file, 'rb') as handle:
    svd = pickle.load(handle)

# #############################################################################
# Do the actual clustering
ward = AgglomerativeClustering(n_clusters=opts.n_clusters,
                               linkage='ward', affinity='euclidean', connectivity=None,
                               # memory=None, compute_full_tree=False
                               )

logging.info("Compute unstructured hierarchical clustering with %s" % ward)
t0 = time()
ward.fit(X)
# Save the fitted clustering for later use
model_file = opts.interim + '{0}-{1}-{2}-{3}-model_hierarchical.pickle'.format(
    opts.size, opts.n_clusters, opts.min_df, opts.max_df)
with open(model_file, 'wb') as handle:
    pickle.dump(ward, handle)
logging.info("  Done in %0.3fs" % (time() - t0))

# Calculate metrics
t0 = time()
silhouette_list = []

logging.info("  Silhouette Coefficient: %0.3f"
             % metrics.silhouette_score(X, ward.labels_, sample_size=1000))
X_to_CH = X if opts.n_components else X.toarray()
logging.info("  Calinski-Harabasz Index: %0.3f"
             % metrics.calinski_harabasz_score(X, ward.labels_))
# note S_Dbw increases metrics calculation time by 150 %
logging.info("  S_Dbw validity index: %0.3f"
             % S_Dbw(X, ward.labels_, alg_noise='bind', centr='mean',
                     metric='euclidean'))
if opts.baseline:
    truth_file = '../models/groundtruth_labels_final.csv'
    labels = pd.read_csv(truth_file, index_col=0).values[:,0].tolist()
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
    # logging.info("   DEBUG: sorted_ctms: {0}".format(sorted_ctms[i, :-16:-1]))
    # logging.info("   DEBUG: top_for_i: {0}".format(top_for_i))
    # logging.info("   DEBUG: terms: {0}".format(len(terms)))
    term_str = ' '.join([terms[j] for j in top_for_i])
    logging.info("  Cluster {0}: {1}".format(i, term_str))
logging.info("  Done in %fs" % (time() - t0))


# Sample of publications per cluster
logging.info('Sample of publications per cluster:')
t0 = time()
sample_max = 5
with open(opts.preproc_file, 'rb') as handle:
    data = pickle.load(handle)
for k in range(opts.n_clusters):
    cluster_k_pubs = [d for (d, l) in zip(data, ward.labels_) if l == k]
    sample_size = sample_max if len(cluster_k_pubs) > sample_max else len(cluster_k_pubs) - 1
    pubs_sample = random.sample(cluster_k_pubs, sample_size)
    logging.info('  Cluster {0}:'.format(k))
    # FIXME: Tie publication's info columns to available fields
    for p in pubs_sample:
        pub_info = '          ' + p['title'] + '|'
        pub_info += p['discipline'] if p.get('discipline') else ''
        logging.info(pub_info)
logging.info("  Done in %fs" % (time() - t0))

# Plot silhouette
plot_silh = False
if plot_silh:
    t0 = time()
    plot_silhouette(X, ward.labels_, opts.n_clusters, 'hierarchical', highlight=[17,120,137])  # [27,66,163]
    logging.info("Silhouette plot in %fs" % (time() - t0))

logging.info("Total running time: %fs" % (time() - t_total))