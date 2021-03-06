# encoding: utf-8
##################################################################
# Step 1: Preprocess data for hand picked disciplines
#
# Usage: python preprocessing.py <label>
##################################################################
import argparse
import pickle
import re

filename = '../data/raw/SuomiRyväsData2000'
exclude_file = '../models/groundtruth_exclude.txt'
preproc_file = '../data/baseline/interim/groundtruth-preproc_CS-AI-IS_CN.pickle'
preproc_txtfile = '../data/baseline/interim/groundtruth-preproc_CS-AI-IS_CN.txt'

opening_line = r'[ ]{,11}\d{,12} (?P<identifier>\d{15})'  # Note here is re.match(opening_line).group('identifier')
journal_line = r'Lehti:\s*(.*)\s*'
issn_line = r'ISSN:\s*(.*)\s*'
discipline_line = r'Ala:\s*(.*)\s*'
year_line = r'Ilmestymisvuosi:\s*(\d{1,4})\s*'
title_line = r'Otsikko:\s*(.*)\s*'
abstract_line = r'Abstrakti'
keyword_publisher_line = r'Avainsana \(KeywordPlus\):\s*(.*)\s*'
keyword_line = r'Avainsana \(tekijät\):\s*(.*)\s*'
reference_line = r'Lähde:\s*(.*)\s*'
closing_line = r' \* '
term_sep = r',*\s+[:&]*\s*'
term_rep = '_'
ref_sep = r'(\d+) /.*'
ref_rep = '\\1'
ref_clean = r',*\/*\s+(\/*\s*)*'
discipline_1 = 'COMPUTER_SCIENCE_ARTIFICIAL_INTELLIGENCE'
discipline_2 = 'COMPUTER_SCIENCE_INFORMATION_SYSTEMS'
discipline_9 = 'CLINICAL_NEUROLOGY'

# Read the parameters for a run.
used_fields = ['id', 'journal', 'issn', 'discipline', 'year', 'title', 'abstract', 'keyword_publisher', 'keyword',
               'reference']
parser = argparse.ArgumentParser(description='Pre-process raw data')
parser.add_argument('used_fields', metavar='F', type=str, nargs='*',
                   help='fields kept while pre-processing')
parser.add_argument('--b_size', dest='b_size', type=int, required=True,
                   help='size of the batch to pre-process')
parser.add_argument('--filter', dest='filtering', action='store_true',
                   help='filter bad data out if given')
args = parser.parse_args()


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

            m = re.match(journal_line, line)
            if m and 'journal' in used_fields:
                current['journal'] = m.group(1)
                continue

            m = re.match(issn_line, line)
            if m and 'issn' in used_fields:
                current['issn'] = m.group(1)
                continue

            # FIXME: Discipline parsed A & B -> A_&_B but split perhaps in vectorizer.
            m = re.match(discipline_line, line)
            if m and 'discipline' in used_fields:
                term = re.sub(term_sep, term_rep, m.group(1))
                value = current.get('discipline')
                if value:
                    current['discipline'] = ' '.join([value, term])
                else:
                    current['discipline'] = term
                continue

            m = re.match(year_line, line)
            if m and 'year' in used_fields:
                current['year'] = m.group(1)
                continue

            m = re.match(title_line, line)
            if m and 'title' in used_fields:
                current['title'] = m.group(1)
                continue

            m = re.match(abstract_line, line)
            if m and 'abstract' in used_fields:
                current['abstract'] = f.readline()
                continue

            m = re.match(keyword_publisher_line, line)
            if m and 'keyword_publisher' in used_fields:
                term = re.sub(term_sep, term_rep, m.group(1))
                value = current.get('keyword_publisher')
                if value:
                    current['keyword_publisher'] = ' '.join([value, term])
                else:
                    current['keyword_publisher'] = term
                continue

            m = re.match(keyword_line, line)
            if m and 'keyword' in used_fields:
                term = re.sub(term_sep, term_rep, m.group(1))
                value = current.get('keyword')
                if value:
                    current['keyword'] = ' '.join([value, term])
                else:
                    current['keyword'] = term
                continue

            m = re.match(reference_line, line)
            if m and 'reference' in used_fields:
                ref = re.sub(ref_sep, ref_rep, m.group(1)).strip(' /')
                ref = re.sub(ref_clean, ref_rep, ref)
                value = current.get('reference')
                if value:
                    current['reference'] = ' '.join([value, ref])
                else:
                    current['reference'] = ref
                continue

            m = re.match(closing_line, line)
            if m:
                # print('discipline: {0}'.format(current.get('discipline')))
                if current.get('discipline') \
                        and (discipline_1 in current['discipline']
                             or discipline_2 in current['discipline']
                             or discipline_9 in current['discipline']):
                    datasets.append(current.copy())
                current = {}
                k += 1
            if k >= batch_size:
                break
    return datasets

# print("Prepocessing data with fields: {0}".format(used_fields))  # debug
data = dataset(batch_size=args.b_size)

# Filter excluded data
filtered_data = []
if args.filtering:
    with open(exclude_file, 'r') as ef:
        excluded = [int(line.rstrip('\n')) for line in ef]
    for (i, d) in enumerate(data, start=1):
        # print('i: {0}, data: {1}'.format(i, d))  # debug
        if i in excluded:
            # print('Excluded: i: {0}, title: {1}'.format(i, d['title']))  # debug
            continue
        else:
            filtered_data.append(d)
else:
    filtered_data = data

# Write pre-processed data in file and in text file
# print(filtered_data[5:7])  # debug
with open(preproc_file, 'wb') as handle:
    pickle.dump(filtered_data, handle)
with open(preproc_txtfile, 'w') as handle:
    for (i, d) in enumerate(filtered_data, start=1):
        for field in ['title', 'abstract', 'keyword', 'keyword_publisher', 'journal', 'discipline']:
            if d.get(field):
                handle.write('{0}  {1}: '.format(i, field) + d[field] + '\n')
        handle.write('\n\n')
