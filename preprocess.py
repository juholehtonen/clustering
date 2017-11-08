# encoding: utf-8
##################################################################
# Step 1: Preprocess data
#
# Usage: python preprocessing.py <label>
##################################################################
import cPickle
import pprint
import re
import sys

filename = '../data/SuomiRyväsData2000'

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
closing_line = r' \* '

# Read the parameters for a run.
b_size = int(sys.argv[1])
# used_fields = ['id', 'issn', 'title', 'abstract', 'keyword_publisher', 'keyword', 'reference']
used_fields = ['title', 'keyword', 'reference']
if len(sys.argv) > 2:
    used_fields = sys.argv[2].split(',')


def dataset(batch_size):
    datasets = []
    current = {}
    k = 0
    with open(filename) as f:
        for line in f:
            m = re.match(opening_line, line)
            if m and 'id' in used_fields:
                current['id'] = m.group('identifier')
                continue

            m = re.match(issn_line, line)
            if m and 'issn' in used_fields:
                current['issn'] = m.group(1)
                continue

            #FIXME: Discipline parsed A & B -> A_&_B but split perhaps in vectorizer.
            m = re.match(discipline_line, line)
            if m:
             value = current.get('discipline')
             if value:
                 current['discipline'] = ' '.join([value, m.group(1).replace(' & ', '_&_')])
             else:
                 current['discipline'] = m.group(1).replace(' & ', '_&_')
                 continue

            # Note: remember to parse two character words too 'chromosome arm 3p'
            m = re.match(title_line, line)
            if m and 'title' in used_fields:
                current['title'] = m.group(1)
                continue

            # Note: remember to parse two character words too 'chromosome arm 3p'
            m = re.match(abstract_line, line)
            if m and 'abstract' in used_fields:
                current['abstract'] = f.next()
                continue

            m = re.match(keyword_publisher_line, line)
            if m and 'keyword_publisher' in used_fields:
                value = current.get('keyword_publisher')
                if value:
                    current['keyword_publisher'] = ' '.join([value,
                                                             m.group(1).replace(' ', '_')])
                else:
                    current['keyword_publisher'] = m.group(1).replace(' ', '_')
                continue

            m = re.match(keyword_line, line)
            if m and 'keyword' in used_fields:
                value = current.get('keyword')
                if value:
                    current['keyword'] = ' '.join([value, m.group(1).replace(' ', '_')])
                else:
                    current['keyword'] = m.group(1).replace(' ', '_')
                continue

            m = re.match(reference_line, line)
            if m and 'reference' in used_fields:
                ref = re.sub('(\d+) /.*', '\\1', m.group(1))
                value = current.get('reference')
                if value:
                    current['reference'] = ' '.join([value, ref.replace(' ', '_')])
                else:
                    current['reference'] = ref.replace(' ', '_')
                continue

            m = re.match(closing_line, line)
            if m:
                datasets.append(current.copy())
                current = {}
                k += 1
            if k >= batch_size:
                break
    return datasets

data = dataset(batch_size=b_size)
with open('../data/{0}-preprocessed.txt'.format(str(b_size)), 'w') as handle:
    cPickle.dump(data, handle)
with open('../data/preprocessed-preview.txt', 'w') as handle:
    handle.write(pprint.pformat(data[:5]))