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
import scipy.sparse
import sys
from time import time


from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from lemmatizer import NLTKPreprocessor
from utils import (GeneralExtractor,
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
op.add_option("--n-features",
              dest="n_features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--no-minibatch",
              dest="minibatch", action="store_false", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              dest="use_idf", action="store_false", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--max-df",
              dest="max_df", type=float, default=0.1,
              help="TfidfVectorizer's max_df parameter")
op.add_option("--verbose",
              dest="verbose", action="store_true", default=False,
              help="Print progress reports inside k-means algorithm.")
op.add_option("--source",
              dest="source", type="string",
              help="Source file to process")
op.add_option("--interim",
              dest="interim", type="string",
              help="Interim folder")
op.add_option("--out",
              dest="out", type="string",
              help="Output dircetory for results")


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
logging.info('#' * 17 + ' Starting vectorization ' + '#' *17)

# #############################################################################
# Load data from the previous step
logging.info("Loading pre-processed data")
with open(opts.source, 'rb') as handle:
    data = pickle.load(handle)

stopwords_ext = list(set(ENGLISH_STOP_WORDS).union(stopwords.words('english')))
stopwords_ext += ['reserved', 'rights', 'science', 'elsevier', '2000']

logging.info("Extracting features from the training dataset using a sparse vectorizer")
min_df = 2

t0 = time()
vectorizer = TfidfVectorizer(max_df=opts.max_df,
                             max_features=opts.n_features,
                             min_df=min_df,
                             # stop_words=stopwords_ext,
                             use_idf=opts.use_idf,
                             # vocabulary=None,
                             tokenizer=lambda x: x,
                             # tokenizer=LemmaTokenizer,
                             # preprocessor=None,
                             lowercase=False,
                             analyzer='word')

vectrzr = make_pipeline(GeneralExtractor(fields=opts.fields.split(',')),
# FIXME: Onko tässä tokenointi (vektorisoijan sisällä) ja lemmatisointi (NLTKPreprocessor) väärinpäin?
# FIXME: Tokenointi itseasiassa lemmatisoijan yhteydessä
                        NLTKPreprocessor(stopwords=stopwords_ext),
                        vectorizer)
X = vectrzr.fit_transform(data)

# Save vectorized data and the vectorizer for the next step
vctd_filename = '{0}-{1}-{2}-{3}-vectorized.npz'.format(opts.size, min_df, opts.max_df, opts.n_features)
scipy.sparse.save_npz(opts.interim + vctd_filename, X)
terms = vectorizer.get_feature_names()
with open(opts.interim + '{0}-vectorizer_feature_names.pickle'.format(str(opts.size)), 'wb') as handle:
    pickle.dump(terms, handle)

logging.info('  Feature extraction steps: {0}'.format([s[0] for s in vectrzr.steps]))
logging.info('  TfidfVectorizer, max_df: {0}, min_df: {1}, max_features: {2}, n_stopwords: {3}'
             .format(opts.max_df, min_df, opts.n_features, len(stopwords_ext)))
logging.info('  Vectorizer tokenizer: {0}'.format(vectorizer.tokenizer.__class__))
logging.info('  Pre-tokenizer: {0}'.format(vectrzr.steps[1][0]))
logging.info("  n_samples: %d, n_features: %d" % X.shape)
logging.info("  Total discarded terms, cut by min_df and max_df: {0}".format(len(vectorizer.stop_words_) - len(stopwords_ext)))
logging.info("  Done in {0}".format((time() - t0)))
