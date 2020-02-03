# encoding: utf-8
##################################################################
# Step 1: Preprocess data
#
# Usage: python preprocessing.py <label>
##################################################################
import logging
import pickle
import pprint
import re
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M",
                    filename='../log/preprocess.log')

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
term_sep = r',*\s+[:&]*\s*'   # ',,,  :&::   '
term_repl = '_'   # replacement for `term_sep`
ref_sep = r'(\d+) /.*'
ref_repl = '\\1'   # replacement for `ref_sep`
ref_clean = r',*\/*\s+(\/*\s*)*'


# Read the parameters for a run.
sample_size = int(sys.argv[1])
used_fields = ['id', 'journal', 'issn', 'discipline', 'year', 'title', 'abstract', 'keyword_publisher', 'keyword',
               'reference']
# if len(sys.argv) > 2:
#     used_fields = sys.argv[2].split(',')

# filename = '../data/raw/SuomiRyväsData2000'

def get_data(batch_size):
    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        return False

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

            # FIXME: Discipline parsed 'A & B C' -> 'A_B_C' but split perhaps in vectorizer.
            m = re.match(discipline_line, line)
            if m and 'discipline' in used_fields:
                term = re.sub(term_sep, term_repl, m.group(1))
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

            # We assume the whole abstract is the next line (only)
            m = re.match(abstract_line, line)
            if m and 'abstract' in used_fields:
                current['abstract'] = f.readline()
                continue

            m = re.match(keyword_publisher_line, line)
            if m and 'keyword_publisher' in used_fields:
                term = re.sub(term_sep, term_repl, m.group(1))
                value = current.get('keyword_publisher')
                if value:
                    current['keyword_publisher'] = ' '.join([value, term])
                else:
                    current['keyword_publisher'] = term
                continue

            m = re.match(keyword_line, line)
            if m and 'keyword' in used_fields:
                term = re.sub(term_sep, term_repl, m.group(1))
                value = current.get('keyword')
                if value:
                    current['keyword'] = ' '.join([value, term])
                else:
                    current['keyword'] = term
                continue

            m = re.match(reference_line, line)
            if m and 'reference' in used_fields:
                ref = re.sub(ref_sep, ref_repl, m.group(1)).strip(' /')
                ref = re.sub(ref_clean, ref_repl, ref)
                value = current.get('reference')
                if value:
                    current['reference'] = ' '.join([value, ref])
                else:
                    current['reference'] = ref
                continue

            m = re.match(closing_line, line)
            if m:
                datasets.append(current.copy())
                current = {}
                k += 1
            if k >= batch_size:
                break
    return datasets


logging.info("Pre-processing data with fields: {0}".format(used_fields))
data = get_data(batch_size=sample_size)
with open('../data/interim/{0}-preprocessed.pickle'.format(str(sample_size)), 'wb') as handle:
    pickle.dump(data, handle)
with open('../data/interim/preprocessed-preview.txt', 'w') as handle:
    handle.write(pprint.pformat(data[:5]))