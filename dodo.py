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
samples = [400]

imagefile = '../img/{0}-{1}-{2}-plot.png'

# def task_preprocess_tiny():
#     """Step 1: preprocess data"""
#     size = 20
#     return {
#         #'name': 'size: {0}'.format(size),
#         'file_dep': ['preprocess.py'],
#         'targets': ['../data/%s-preprocessed.txt' % size],
#         'actions': ['python preprocess.py %s' % size],
#     }
        
        
def task_preprocess_small():
    """Step 1: preprocess data"""
    #size = 200
    size = samples[0]
    return {
        #'name': 'size: {0}'.format(size),
        'file_dep': ['preprocess.py'],
        'targets': ['../data/%s-preprocessed.txt' % size],
        'actions': ['python preprocess.py %s' % size],
    }


def task_analyze_mds():
    """Step 2: cluster data"""
    size = samples[0]
    label = 'mds'
    return {
        #'name': label,
        'file_dep': ['analyse_mds.py',
                     '../data/%s-preprocessed.txt' % size],
        'targets': [imagefile.format(label, vect, size)],
        'actions': ['python analyse_mds.py {0} {1}'.format(size, vect)],
    }


def task_show_images():
    """Step 3: view saved images"""
    size = samples[0]
    for label in labels:
        yield {
            'name': label,
            'file_dep': [imagefile.format(label, vect, size)],
            'actions': ['display ' + imagefile.format(label, vect, size)]
        }
