# encoding: utf-8
##################################################################
# Master plumbing to run analysis pipeline
#
# Usage: doit -n 7
##################################################################
import nltk
from pathlib import Path

# labels = ['mds', 'baseline']
# labels = ['mds']
# vect = 'countvectorizer'
vect = 'tfidfvectorizer'
# Define metadata fields used in clustering analysis
analysis_fields = 'title,abstract,keyword,keyword_publisher'
samples = [200, 400, 6000, 12000, 48000]
# 220 = roughly number of disciplines in 6000 first datasets
# n_clusters = [4, 32, 64, 128, 220]
cluster_range = list(range(500,20,-9)) + list(range(20,2,-3))
# cluster_range = list(range(265,181,-7))
# cluster_range.extend([150, 100, 30])
# cluster_range = [3]
# cluster_range = range(260, 250, -3)
n_components = [800, 300, 200, 40]
n_features = [10000, 50000]
df_min = 2
df_max = 0.1

size = samples[4]
# k = n_clusters[1]
n_comp = n_components[0]
n_comp_str = n_comp if n_comp < 600 else '-'
n_feat = n_features[1]

input_dir = '../data/raw/'
input_filename = 'SuomiRyväsData2000-3'
input_file = input_dir + input_filename
interim_dir = '../data/interim/'
#preprocess_view = '{size}-preprocessed.txt'
# preproc_file_name = '{0}-preprocessed.pickle'.format(size)
preproc_file_name = '{0}-{1}-preproc.pickle'.format(input_filename, size)
preproc_file = interim_dir + preproc_file_name
vectorized_file_name = '{0}-{1}-{2}-{3}-vectorized.npz'.format(size, df_min, df_max, n_feat)
vectorized_file = interim_dir + vectorized_file_name
reduced_file_name = '{0}-{1}-{2}-{3}-{4}-vect-reduced.pickle'.format(size, df_min, df_max, n_feat, n_comp)
reduced_file = interim_dir + reduced_file_name
svd_file_name = '{0}-{1}-{2}-{3}-{4}-svd.pickle'.format(size, df_min, df_max, n_feat, n_comp)
svd_file = interim_dir + svd_file_name
results_dir = '../data/processed/'
#results_tmpl = '{size}-{k}-{ncomp}-{algorithm}-results.txt'
image_file_name = '{0}-{1}-{2}-{3}-plot.png'
image_file = results_dir + image_file_name



def task_init():
    """
    Initialize NLTK corpora etc.
    """
    usrhome = str(Path.home())

    def init_nltk(corpus):
        nltk.download(corpus)

    corpora = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet']
    return {
        'targets': [usrhome + '/nltk_data/corpora/wordnet.zip'],
        'actions': [(init_nltk, [corpora])],
        # force doit to always mark the task as up-to-date (unless target removed)
        'uptodate': [True]
    }


def task_preprocess_small():
    """Step 1: preprocess data"""
    output_file = preproc_file
    return {
        #'name': 'size: {0}'.format(size),
        'file_dep': ['preprocess.py',
                     input_file],
        'targets': [preproc_file],
        'actions': ['python preprocess.py {0} {1} {2}'.format(size, input_file, output_file)],
    }


def task_vectorize():
    """Step 2: vectorize data"""
    options = '--size {0} --n-features {1} --fields {2} --max-df {3} --min-df {4} --source {5} --interim {6} --out {7}'\
              .format(size, n_feat, analysis_fields, df_max, df_min, preproc_file, interim_dir, results_dir)
    return {
        'file_dep': ['vectorize.py',
                     preproc_file],
        'targets': [vectorized_file],
        'actions': ['python vectorize.py {0}'.format(options)],
    }

def task_reduce():
    """Step 3: reduce the dimensionality of vectorized data"""
    options = '--size {0} --n-features {1} --max-df {2} --min-df {3} --lsa {4} --source {5} --interim {6} --out {7}'\
              .format(size, n_feat, df_max, df_min, n_comp, vectorized_file, interim_dir, results_dir)
    return {
        'file_dep': ['reduce_lsa.py',
                     vectorized_file],
        'targets': [reduced_file],
        'actions': ['python reduce_lsa.py {0}'.format(options)],
    }

# def task_analyze_mds():
#     """Step 1.5: 2D embedding of data"""
#     pass
#     size = samples[0]
#     label = 'mds'
#     return {
#         #'name': label,
#         'file_dep': ['analyse_mds.py',
#                      '../data/interim/%s-preprocessed.txt' % size],
#         'targets': [image_file.format(size, vect, label, 2)],
#         'actions': ['python analyse_mds.py {0} {1}'.format(size, vect)],
#     }


# def task_analyze_mini_k_means():
#     """Step 3: cluster data"""
#     # label = 'minikm'
#     for n in cluster_range:
#         options = '--size {0} --n-clusters {1} --no-minibatch --lsa {2} --n-features {3} --fields {4} --source {5} --interim {6} --out {7}'\
#               .format(size, n, n_comp, n_feat, analysis_fields, preproc_file, interim_dir, results_dir)
#         yield {
#             'name': ' k: {0}'.format(n),
#             'file_dep': ['analyse_mini-k-means.py',
#                          vectorized_file],
#             'targets': [image_file.format(size, n, n_comp_str, 'kmeans')],
#             'actions': ['python analyse_mini-k-means.py {0}'.format(options)],
#         }


def task_analyze_hierarchical():
    """Step 3: cluster data"""
    imagefile = results_dir + '{0}-{1}-{2}-{3}-plot.png'

    for n in cluster_range:
        options = '--size {0} --n-clusters {1} --lsa {2} --n-features {3}' \
                  ' --fields {4} --source {5} --preproc-file {6} --svd-file {7} ' \
                  '--interim {8} --out {9} --min-df {10} --max-df {11}' \
            .format(size, n, n_comp, n_feat, analysis_fields, reduced_file, preproc_file,
                    svd_file, interim_dir, results_dir, df_min, df_max)
        yield {
            'name': ' k: {0}'.format(n),
            'file_dep': ['analyse_hierarchical.py',
                         preproc_file,
                         reduced_file],
            'targets': [imagefile.format(size, n, n_comp_str, 'hierarchical')],
            'actions': ['python analyse_hierarchical.py {0}'.format(options)],
        }


# def task_show_images():
#     """Step 3: view saved images"""
#     for label in labels:
#         yield {
#             'name': label,
#             'file_dep': [image_file.format(size, k, n_comp_str, 'Hierarchical')],
#             'actions': ['display ' + image_file.format(size, k, n_comp_str, 'Hierarchical')]
#         }
