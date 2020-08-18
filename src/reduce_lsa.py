# coding=UTF8
##################################################################
# Step 3: Reduce data dimensionality with LSA
#
# Usage: python reduce_lsa.py <opts>
##################################################################
from __future__ import print_function

import logging
from optparse import OptionParser
import pickle
import scipy.sparse
import sys
from time import time

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (Normalizer,
                                   FunctionTransformer)

# parse commandline arguments
op = OptionParser()
op.add_option("--size",
              dest="size", type="int", default=400,
              help="Size of the preprocessed data to be used.")
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--max-df", dest="max_df", type=float, default=0.1,
              help="TfidfVectorizer's max_df parameter")
op.add_option("--min-df",
              dest="min_df", type=int, default=2,
              help="TfidfVectorizer's min_df parameter")
op.add_option("--source",
              dest="source", type="string",
              help="Source file to process")
op.add_option("--interim",
              dest="interim", type="string",
              help="Interim folder")
op.add_option("--out",
              dest="out", type="string",
              help="Output directory for results")

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

# Define log file name and start log
results_filename = opts.out + '{0}-{1}-{2}-{3}-{4}-reduce_lsa.log'.format(opts.size, opts.min_df, opts.max_df, opts.n_features, opts.n_components)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M",
                    filename=results_filename)
logging.info('#' * 12 + ' Starting dimensionality reduction ' + '#' *12)

# #############################################################################
# Load data from the previous step
logging.info("Loading vectorized data")
t_total = time()
vect_file = opts.source
logging.info("   vectorized data file: {0}".format(vect_file))
X = scipy.sparse.load_npz(vect_file)

# #############################################################################
# Dimensionality reduction
logging.info("Performing dimensionality reduction using LSA")
logging.info("  Number of input features: {0}".format(X.shape[1]))
t0 = time()
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(opts.n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

explained_variance = svd.explained_variance_ratio_.sum()
logging.info("  Explained variance of the SVD step: {0}% with {1} components".format(
    int(explained_variance * 100), opts.n_components))
logging.info("  Done in %fs" % (time() - t0))

# Save reduced dimensionality data and SVD for the next step
reduced_filename = '{0}-{1}-{2}-{3}-{4}-vect-reduced.pickle'.format(opts.size, opts.min_df, opts.max_df, opts.n_features, opts.n_components)
svd_filename = '{0}-{1}-{2}-{3}-{4}-svd.pickle'.format(opts.size, opts.min_df, opts.max_df, opts.n_features, opts.n_components)
logging.info("  reduced data file: {0}".format(reduced_filename))
logging.info("  SVD file: {0}".format(svd_filename))
with open(opts.interim + reduced_filename, 'wb') as handle:
    pickle.dump(X, handle)
with open(opts.interim + svd_filename, 'wb') as handle:
    pickle.dump(svd, handle)
