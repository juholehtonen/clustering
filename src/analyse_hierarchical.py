# coding=UTF8
##################################################################
# Step 2: Cluster data with hierarchical Ward's method.
#
# Usage: python analyse_hierarchical.py <size>
##################################################################
from __future__ import print_function

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (Normalizer,
                                   FunctionTransformer)
from sklearn import metrics

from sklearn.cluster import AgglomerativeClustering

import pickle
import logging
from nltk.corpus import stopwords
import numpy as np
from optparse import OptionParser
import pandas as pd
import random
import sys
from time import time

from utils import GeneralExtractor, plot_silhouette, LemmaTokenizer
from lemmatizer import NLTKPreprocessor


# parse commandline arguments
op = OptionParser()
op.add_option("--file",
              dest="inputfile", type="string",
              help="Data file to run analysis with.")
op.add_option("--fields",
              dest="fields", type="string",
              help="Metadata fields to run analysis with.")
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--max-df", dest="max_df", type=float, default=0.1,
              help="TfidfVectorizer's max_df parameter")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")
op.add_option("--size",
              dest="size", type="int", default=400,
              help="Size of the preprocessed data to be used.")
op.add_option("--n-clusters",
              dest="n_clusters", type="int", default=16,
              help="Number of clusters to be used.")


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

# Display progress logs on stdout
results_filename = '../data/processed/'
#'--size {0} --n-clusters {1} --lsa {2} --n-features {3} --fields {4}'\
#              .format(size, k, n_comp, n_feat, analysis_fields)
for o in [opts.size, opts.n_clusters]:
    results_filename = results_filename + str(o) + '-'
if opts.n_components:
    results_filename = results_filename + str(opts.n_components) + '-'
results_filename += 'hierarchical-results.txt'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M",
                    filename=results_filename)
logging.info('#' * 24 + ' Starting... ' + '#' *24)

# #############################################################################
# Load data from the previous step
logging.info("Loading prepocessed data")
with open(opts.inputfile, 'rb') as handle:
  data = pickle.load(handle)


#labels = dataset.target
#true_k = np.unique(labels).shape[0]
stopwords_ext = list(set(ENGLISH_STOP_WORDS).union(stopwords.words('english')))
stopwords_ext += ['reserved', 'rights', 'science', 'elsevier', '2000']

logging.info("Extracting features from the training dataset using a sparse vectorizer")
min_df = 2
t_total = t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words=stopwords_ext, alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words=stopwords_ext,
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=opts.max_df,
                                 max_features=opts.n_features,
                                 min_df=min_df,
                                 stop_words=stopwords_ext,
                                 use_idf=opts.use_idf,
                                 vocabulary=None,
                                 tokenizer=lambda x: x,
                                 preprocessor=None,
                                 lowercase=False)

vectrzr = make_pipeline(GeneralExtractor(fields=opts.fields.split(',')),
                        NLTKPreprocessor(),
                        vectorizer,
                        FunctionTransformer(lambda x: x.todense(), accept_sparse=True))
X = vectrzr.fit_transform(data)
logging.info("done in {0}".format((time() - t0)))
logging.info('Feature extraction steps: {0}'.format([s[0] for s in vectrzr.steps]))
logging.info('TfidfVectorizer, max_df: {0}, min_df: {1}, max_features: {2}, n_stopwords: {3}'
             .format(opts.max_df, min_df, opts.n_features, len(stopwords_ext)))
logging.info('Vectorizer tokenizer: {0}'.format(vectorizer.tokenizer.__class__))
logging.info('Pre-tokenizer: {0}'.format(vectrzr.steps[1][0]))
logging.info("n_samples: %d, n_features: %d" % X.shape)
logging.info("total discarded terms: {0}".format(len(vectorizer.stop_words_) - len(stopwords_ext)))

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

    logging.info("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    logging.info("Explained variance of the SVD step: {0}% with {1} components".format(
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
logging.info("done in %0.3fs" % (time() - t0))

t0 = time()
logging.info("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, ward.labels_, sample_size=1000))

logging.info("Calinski-Harabasz Index: %0.3f"
      % metrics.calinski_harabasz_score(X, ward.labels_))
logging.info("Metrics calculated in %fs" % (time() - t0))

if not opts.use_hashing:
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
    terms = vectorizer.get_feature_names()
    for i in range(opts.n_clusters):
        top_for_i = sorted_ctms[i, ::-1][:15]
        term_str = ' '.join([terms[j] for j in top_for_i])
        logging.info("Cluster {0}: {1}".format(i, term_str))
    logging.info("done in %fs" % (time() - t0))

logging.info('Sample of publications per cluster:')
t0 = time()
for i in range(opts.n_clusters):
    pubs = [d for (d, l) in zip(data, ward.labels_) if l == i]
    sample_size = 15 if len(pubs) > 15 else len(pubs)-1
    pubs_sample = random.sample(pubs, sample_size)
    logging.info('Cluster {0}:'.format(i))
    for p in pubs_sample:
        logging.info('          ' + p['title'][:80]
                     + (80 - len(p['title']))*' ' + '|'
                     + p['discipline'][:30])
logging.info("done in %fs" % (time() - t0))

t0 = time()
plot_silhouette(X, ward.labels_, opts.n_clusters, 'Hierarchical')
logging.info("Silhouette plot in %fs" % (time() - t0))
logging.info("Total running time: %fs" % (time() - t_total))