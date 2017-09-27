# encoding: utf-8
##################################################################
# Master plumbing to run analysis pipeline
#
# Usage: doit
##################################################################

# labels = ['mds', 'baseline']
labels = ['baseline']
samples = [20, 60, 200]


def task_preprocess():
    """Step 1: pre process data"""
    for size in samples:
        yield {
            'name': 'size: {0}'.format(size),
            'file_dep': ['preprocess.py'],
            'targets': ['data/%s-preprocessed.txt' % size],
            'actions': ['python preprocess.py %s' % size],
        }


def task_analyze():
    """Step 2: cluster data"""
    size = samples[0]
    for label in labels:
        yield {
            'name': label,
            'file_dep': ['analyse_mds.py',
                         'data/%s-preprocessed.txt' % size],
            'targets': ['%s-plot.png' % label],
            'actions': ['python analyse_mds.py %s' % label],
        }


def task_show_images():
    """Step 3: view saved images"""
    for label in labels:
        yield {
            'name': label,
            'file_dep': ['%s-plot.png' % label],
            'actions': ['display {0}-plot.png'.format(label)]
        }
