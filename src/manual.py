# encoding: utf-8
##################################################################
# 'Doit' file to run analysis on manually annotated dataset
#
# Usage: doit -f manual.py
##################################################################
import nltk
from pathlib import Path

labels = ['mds']
vect = 'tfidfvectorizer'
# Define metadata fields used in clustering analysis
analysis_fields = 'title,abstract,keyword'
samples = [200, 400, 2000, 6000, 12000]

n_clusters = [4, 32, 64, 128, 220]
cluster_range = range(2, 3)
n_components = [800, 300, 200, 40]
n_features = [10000]
df_min = 2
df_max = 0.1

size = samples[4]
k = n_clusters[1]
n_comp = n_components[0]
n_comp_str = n_comp if n_comp < 600 else '-'
n_feat = n_features[0]


interim = '../data/baseline/interim/'
results = '../data/baseline/results/'
ppfile = 'groundtruth-preproc_CS-AI-IS_CN.pickle'
srcfile = interim + ppfile

results_tmpl = '{size}-{k}-{ncomp}-{algorithm}-results.txt'
preprocess_tmpl = '{size}-preprocessed.pickle'
preprocess_view = '{size}-preprocessed.txt'
preprocess_file = '../data/interim/{0}-preprocessed.pickle'.format(size)
# preprocess_file = '../data/interim/ground-truth_CS-AI-IS-CN_preprocessed.pickle'


# def task_preprocess_groundtruth():
#     """ Step 1: preprocess data """
#     size = 20000
#     return {
#         # 'name': 'size: {0}'.format(size),
#         'file_dep': ['preprocess_groundtruth.py'],
#         'targets': ['../data/baseline/groundtruth-preproc_CS-AI-IS_CN.pickle'],
#         'actions': ['python preprocess_groundtruth.py %s' % size],
#         'verbosity': 2
#     }


def task_init():
    """
    Initialize NLTK corpora etc.
    """
    usrhome = str(Path.home())

    def init_nltk(corpus):
        nltk.download(corpus)

    corpuses = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet']
    return {
        'targets': [usrhome + '/nltk_data/corpora/wordnet.zip'],
        'actions': [(init_nltk, [corpuses])],
        # force doit to always mark the task as up-to-date (unless target removed)
        'uptodate': [True]
    }


def task_vectorize():
    """Step 2: vectorize data"""
    options = '--size {0} --n-clusters {1} --no-minibatch --lsa {2} --n-features {3} --fields {4} --source {5} --interim {6} --out {7}'\
              .format(size, k, n_comp, n_feat, analysis_fields, srcfile, interim, results)
    return {
        'file_dep': ['vectorize.py', srcfile],
        'targets': [interim + '{0}-{1}-{2}-{3}-vectorized.npz'.format(size, df_min, df_max, n_feat)],
        'actions': ['python vectorize.py {0}'.format(options)],
    }


def task_analyze_mini_k_means():
    """Step 3: cluster data"""
    label = 'minikm'
    imagefile = results + '{0}-{1}-{2}-{3}-plot.png'
    for k in cluster_range:
        options = '--size {0} --n-clusters {1} --no-minibatch --lsa {2} --n-features {3} --fields {4} --source {5} --interim {6} --out {7}'\
              .format(size, k, n_comp, n_feat, analysis_fields, srcfile, interim, results)
        yield {
            'name': ' k: {0}'.format(k),
            'file_dep': ['analyse_mini-k-means.py',
                         interim + '{0}-{1}-{2}-{3}-vectorized.npz'.format(size, df_min, df_max, n_feat)],
            'targets': [imagefile.format(size, k, n_comp_str, 'kmeans')],
            'actions': ['python analyse_mini-k-means.py {0}'.format(options)],
        }


# def task_analyze_hierarchical():
#     """Step 3: cluster data"""
#     options = '--file {5} --n-clusters {1} --lsa {2} --n-features {3} --fields {4} --size {0}'\
#               .format(size, k, n_comp, n_feat, analysis_fields, preprocess_file)
#     return {
#         #'name': label,
#         'file_dep': ['analyse_hierarchical.py',
#                      '%s' % preprocess_file],
#         'targets': [imagefile.format(size, k, n_comp_str, 'hierarchical')],
#         'actions': ['python analyse_hierarchical.py {0}'.format(options)],
#     }


# def task_show_images():
#     """Step 3: view saved images"""
#     for label in labels:
#         yield {
#             'name': label,
#             'file_dep': [imagefile.format(size, k, n_comp_str, 'Hierarchical')],
#             'actions': ['display ' + imagefile.format(size, k, n_comp_str, 'Hierarchical')]
#         }
