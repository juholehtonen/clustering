# encoding: utf-8
##################################################################
# 'Doit' file to run hierarchical clustering on manually annotated dataset
#
# Usage: doit [-n 8] -f do_manual_h.py
##################################################################
import nltk
from pathlib import Path

labels = ['mds']
vect = 'tfidfvectorizer'
# Define metadata fields used in clustering analysis
analysis_fields = 'title,abstract,keyword'
# size: We want to filter whole year data for the three disciplines:
# grep "^Otsikko: " data/raw/SuomiRyväsData200[0-0]|wc -l
size = 12000
baseline = True
filtering = True

n_clusters = [12, 32, 64, 128, 220, 260]
n_components = [800, 300, 90, 40]
n_features = [10000]
df_min = 2
df_max = 0.1


cluster_range = range(2, n_clusters[0]+1)
n_comp = n_components[2]
n_comp_str = n_comp if n_comp < 600 else '-'
n_feat = n_features[0]


interim_dir = '../data/baseline/interim/'
preproc_file_name = 'groundtruth-preproc_CS-AI-IS_CN.pickle'
preproc_file = interim_dir + preproc_file_name
results_dir = '../data/baseline/results/'


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


def task_preprocess_groundtruth():
    """ Step 1: preprocess data """
    opts = '--b_size {0}'.format(size)
    if filtering:
        opts += ' --filter'
    return {
        # 'name': 'size: {0}'.format(size),
        'file_dep': ['preprocess_groundtruth.py'],
        'targets': [preproc_file],
        'actions': ['python preprocess_groundtruth.py {0}'.format(opts)],
        'verbosity': 2
    }


def task_vectorize():
    """Step 2: vectorize data"""
    options = '--size {0} --n-features {1} --fields {2} --max-df {3} --min-df {4} --source {5} --interim {6} --out {7}'\
              .format(size, n_feat, analysis_fields, df_max, df_min, preproc_file, interim_dir, results_dir)
    return {
        'file_dep': ['vectorize.py', preproc_file],
        'targets': [interim_dir + '{0}-{1}-{2}-{3}-vectorized.npz'.format(size, df_min, df_max, n_feat)],
        'actions': ['python vectorize.py {0}'.format(options)],
    }


def task_analyze_hierarchical():
    """Step 3: cluster data"""
    imagefile = results_dir + '{0}-{1}-{2}-{3}-plot.png'
    vectorized_file = interim_dir + '{0}-{1}-{2}-{3}-vectorized.npz'.format(size, df_min, df_max, n_feat)

    for n in cluster_range:
        options = '--size {0} --n-clusters {1} --lsa {2} --n-features {3}' \
                  ' --fields {4} --source {5} --interim {6} --out {7} --baseline {8} --min-df {8} --max-df {9}' \
            .format(size, n, n_comp, n_feat, analysis_fields, preproc_file, interim_dir, results_dir, baseline, df_min, df_max)
        yield {
            'name': ' k: {0}'.format(n),
            'file_dep': ['analyse_hierarchical.py',
                         vectorized_file],
            'targets': [imagefile.format(size, n, n_comp_str, 'hierarchical')],
            'actions': ['python analyse_hierarchical.py {0}'.format(options)],
        }


# def task_show_images():
#     """Step 3: view saved images"""
#     for label in labels:
#         yield {
#             'name': label,
#             'file_dep': [imagefile.format(size, k, n_comp_str, 'Hierarchical')],
#             'actions': ['display ' + imagefile.format(size, k, n_comp_str, 'Hierarchical')]
#         }
