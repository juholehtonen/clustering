# encoding: utf-8
##################################################################
# Step 1: Preprocess data
#
# Usage: python preprocessing.py <label>
##################################################################
import numpy as np
import cPickle
import re
import sys

# Read the label for a run.
b_size = sys.argv[1]

filename = 'data/SuomiRyväsData2000'

opening_line = r'[ ]{,11}\d{,12} (?P<identifier>\d{15})'   # Note here is re.match(opening_line).group('identifier')
journal_line = r'Lehti: (.*)'
issn_line = r'ISSN: (.*)'
discipline_line = r'Ala: (.*)'
year_line = r'Ilmestymisvuosi:[ ]*(\d{1,4})'
title_line = r'Otsikko: (.*)'
abstract_line = r'Abstrakti'
keyword_publisher_line = r'Avainsana \(KeywordPlus\):  (.*)'
keyword_line = r'Avainsana \(tekijät\):  (.*)'
reference_line = r'Lähde: (.*)'
closing_line = r' * '


def dataset(batch_size):
    datasets = []
    data = {'keyword_publisher': '',
            'keyword': '',
            'reference': ''}
            #'discipline': '',}
    k = 0
    with open(filename) as f:
        for line in f:
            m = re.match(opening_line, line)
            if m:
                data['id'] = m.group('identifier')
                continue

            m = re.match(issn_line, line)
            if m:
                data['issn'] = m.group(1)
                continue

            # m = re.match(discipline_line, line)
            # if m:
            #     data['discipline'] = ' '.join([data['discipline'], m.group(1).replace(' & ', '_&_')])
            #     continue

            # Note: remeber to parse two character words too 'chromosome arm 3p'
            m = re.match(title_line, line)
            if m:
                data['title'] = m.group(1)
                continue

            # Note: remeber to parse two character words too 'chromosome arm 3p'
            m = re.match(abstract_line, line)
            if m:
                data['abstract'] = f.next()
                continue

            m = re.match(keyword_publisher_line, line)
            if m:
                data['keyword_publisher'] = ' '.join([data['keyword_publisher'], m.group(1).replace(' ', '_')])
                continue

            m = re.match(keyword_line, line)
            if m:
                data['keyword'] = ' '.join([data['keyword'], m.group(1).replace(' ', '_')])
                continue

            m = re.match(reference_line, line)
            if m:
                ref = re.sub('(\d+) /.*', '\\1', m.group(1))
                data['reference'] = ' '.join([data['reference'], ref.replace(' ', '_')])
                continue

            m = re.match(closing_line, line)
            if m:
                datasets.append(data.copy())
                data['keyword_publisher'] = ''
                data['keyword'] = ''
                data['reference'] = ''
                k += 1
            if k >= batch_size:
                break
    return datasets

data = dataset(batch_size=b_size)
#import pprint
#pprint.pprint(data[:3])
with open('data/{0}-preprocessed.txt'.format(str(b_size)), 'w') as handle:
    cPickle.dump(data, handle)
#np.savetxt('data/{0}-preprocessed.txt'.format(label), data)