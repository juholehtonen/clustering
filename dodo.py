# encoding: utf-8
##################################################################
# Master plumbing to run analysis pipeline
#
# Usage: doit
##################################################################

# labels = ['mds', 'baseline']
labels = ['mds']
#vect = 'countvectorizer'
vect = 'tfidfvectorizer'
# Define metadata fields used in clustering analysis
analysis_fields = 'title,abstract,keyword'
samples = [200, 400, 2000, 6000]
n_clusters = [4, 32, 64, 128]
n_components = [800, 300, 200, 40]
n_features = [20000]

size = samples[3]
k = n_clusters[2]
n_comp = n_components[0]
n_comp_str = n_comp if n_comp < 600 else '-'
n_feat = n_features[0]


results_tmpl = '{size}-{k}-{ncomp}-{algorithm}-results.txt'
preprocess_tmpl = '{size}-preprocessed.pickle'
preprocess_view = '{size}-preprocessed.txt'
preprocess_file = '../data/%s-preprocessed.txt'
preprocess_file = '../data/ground-truth_CS-AI-IS-CN_preprocessed.pickle'
imagefile = '../img/{0}-{1}-{2}-{3}-plot.png'

def task_preprocess_groundtruth():
    """Step 1: preprocess data"""
    size = 20000
    return {
        #'name': 'size: {0}'.format(size),
        'file_dep': ['preprocess_groundtruth.py'],
        'targets': ['../data/ground_truth-preproc_CS-AI-IS_CN.pickle'],
        'actions': ['python preprocess_groundtruth.py %s' % size],
        'verbosity': 2
    }

        
# def task_preprocess_small():
#     """Step 1: preprocess data"""
#     #size = 200
#     return {
#         #'name': 'size: {0}'.format(size),
#         'file_dep': ['preprocess.py'],
#         'targets': ['../data/%s-preprocessed.txt' % size],
#         'actions': ['python preprocess.py {0}'.format(size)],
#     }


# def task_analyze_mds():
#     """Step 1.5: 2D embedding of data"""
#     pass
#     size = samples[0]
#     label = 'mds'
#     return {
#         #'name': label,
#         'file_dep': ['analyse_mds.py',
#                      '../data/%s-preprocessed.txt' % size],
#         'targets': [imagefile.format(size, vect, label, 2)],
#         'actions': ['python analyse_mds.py {0} {1}'.format(size, vect)],
#     }


def task_analyze_mini_k_means():
    """Step 2: cluster data"""
    label = 'minikm'
    options = '--size {0} --n-clusters {1} --no-minibatch --lsa {2} --n-features {3} --fields {4}'\
              .format(size, k, n_comp, n_feat, analysis_fields)
    return {
        #'name': label,
        'file_dep': ['analyse_mini-k-means.py',
                     '../data/%s-preprocessed.txt' % size],
        'targets': [imagefile.format(size, k, n_comp_str, 'kmeans')],
        'actions': ['python analyse_mini-k-means.py {0}'.format(options)],
    }

def task_analyze_hierarchial():
    """Step 2: cluster data"""
    options = '--file {5} --size {0} --n-clusters {1} --lsa {2} --n-features {3} --fields {4}'\
              .format(size, k, n_comp, n_feat, analysis_fields, preprocess_file)
    return {
        #'name': label,
        'file_dep': ['analyse_hierarchical.py',
                     '%s' % preprocess_file],
        'targets': [imagefile.format(size, k, n_comp_str, 'hierarchical')],
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
