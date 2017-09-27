# encoding: utf-8
##################################################################
# Master plumbing to run analysis pipeline
#
# Usage: doit
##################################################################

labels = ['mds', 'baseline']
#labels = ['baseline']
#samples = [20, 60, 200]


def task_preprocess_tiny():
    """Step 1: preprocess data"""
    size = 20
    return {
        #'name': 'size: {0}'.format(size),
        'file_dep': ['preprocess.py'],
        'targets': ['../data/%s-preprocessed.txt' % size],
        'actions': ['python preprocess.py %s' % size],
    }
        
        
def task_preprocess_small():
    """Step 1: preprocess data"""
    size = 200
    return {
        #'name': 'size: {0}'.format(size),
        'file_dep': ['preprocess.py'],
        'targets': ['../data/%s-preprocessed.txt' % size],
        'actions': ['python preprocess.py %s' % size],
    }


def task_analyze_mds():
    """Step 2: cluster data"""
    size = 200
    label = 'mds'
    return {
        #'name': label,
        'file_dep': ['analyse_mds.py',
                     '../data/%s-preprocessed.txt' % size],
        'targets': ['../img/%s-plot.png' % label],
        'actions': ['python analyse_mds.py %s' % label],
    }


def task_show_images():
    """Step 3: view saved images"""
    for label in labels:
        yield {
            'name': label,
            'file_dep': ['../img/%s-plot.png' % label],
            'actions': ['display ../img/{0}-plot.png'.format(label)]
        }
