import os
import time
import spacy
import argparse
import warnings
import pandas as pd
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='The csv file storing the reviews.')
    parser.add_argument('-o', '--output', type=str,
                        help='The output csv file with modifier-aspect pairs.')
    parser.add_argument('-i', '--item_column', type=str, required=True,
                        help='The column storing item IDs.')
    parser.add_argument('-r', '--review_column', type=str, required=True,
                        help='The column storing review text.')
    parser.add_argument('-c', '--chunksize', type=int, default=-1,
                        help='Chunk size to read large input csv files')
    args = parser.parse_args()
    if args.output is None:   # default value if output is not provided
        args.output = os.path.splitext(args.filename)[0] + '_processed.csv'
    return args


def get_extractions_from_reviews(filename, output, item_column,
                                 review_column, chunksize=-1):
    tqdm.pandas()
    print('loading spacy model...')
    nlp = spacy.load('en_core_web_md', disable=["ner"])

    print('loading review dataset...')
    if os.path.exists(output):
        os.remove(output)

    start = time.time()
    if chunksize == -1:
        data = pd.read_csv(filename)
        print('processing review dataset...')
        ext_data = process_chunk(item_column, review_column, nlp, data)
        if ext_data is not None:
            ext_data.to_csv(output, index=None)
    else:
        writer = open(output, 'a')
        for data_chunk in pd.read_csv(filename, chunksize=chunksize):
            print('processing next {} rows of review ' +
                  'dataset...'.format(chunksize))
            ext_data = process_chunk(nlp, data_chunk)
            if ext_data is not None:
                ext_data.to_csv(writer, header=writer.tell() == 0, index=None)
    end = time.time()
    print('Extraction completed in %d seconds' % (end - start))


def process_chunk(item_column, review_column, nlp, data_chunk):
    processed_chunk = \
        data_chunk.progress_apply(
            lambda x: process_row(item_column, review_column, nlp, x), axis=1)
    try:  # throws index errors if stack is empty
        processed_chunk_rows = \
            processed_chunk.apply(pd.Series).stack().reset_index(drop=True)
        ext_data = pd.DataFrame(processed_chunk_rows.tolist())
    except:
        return None
    return ext_data


def process_row(item_column, review_column, nlp, row):
    curr_id = row[item_column]
    review_id = row[review_column]
    extractions = []
    try:
        pairs = find_mod_asp_pairs(nlp(str(row[review_column])))
        extractions = [{"item_id": curr_id, "review_id": review_id,
                        "modifier": str(mod).replace(",", " "),
                        "aspect": str(asp).replace(",", " ")}
                       for (mod, asp) in pairs]
    except ValueError:
        warnings.warn("text is perhaps too long to process. skipping row..")
    return extractions


def find_mod_asp_pairs(doc):
    # see https://github.com/vladsandulescu/phrases for the set of all rules
    extractions = []
    for t in doc:
        extractions += rule1_0(t)
        extractions += rule1(t)
        extractions += rule2and3(t)
        # Rule 4 is not desired as it finds VERBs.
    return extractions


def rule1_0(t):
    # Rule 1: amod(N, A) --> (mod:A, asp:N) -- deprecate due to low precision?
    if (t.pos_ == 'NOUN'):
        for child in t.children:
            if child.dep_ == 'amod' and child.pos_ == 'ADJ':
                return enumerate_pairs(child, t)
    return []


def rule1(t):
    # Rule 1: amod(N, A) --> (mod:A, asp:N) -- deprecated due to low precision
    # Rule 1++: dobj('has', N) & amod(N, A) --> (mod:A, asp:N)
    if (t.pos_ == 'NOUN' and t.dep_ == 'dobj' and
        t.head.text.lower() in ['has', 'have'] and
        t.head.pos_ in ['VERB', 'AUX']):
        for child in t.children:
            if child.dep_ == 'amod' and child.pos_ == 'ADJ':
                return enumerate_pairs(child, t)
    return []


def rule2and3(t):
    # Rule 2: acomp(V, A) & nsubj(V, N) --> (mod:A, asp:N)
    # Rule 3: cop(A, V ) + nsubj(A, N) --> (mod:A, asp:N)
    # Using Spacy, rule 3 is aboslute as it models both with 'acomp'
    if t.pos_ in ['VERB', 'AUX']:
        mod, asp = None, None
        neg = False
        # searching the children
        for child in t.children:
            if child.dep_ == 'acomp' and child.pos_ == 'ADJ':
                mod = child
            elif child.dep_ == 'nsubj' and child.pos_ == 'NOUN':
                asp = child
            elif child.dep_ == 'neg' and child.text.lower() in ['not', "n't"]:
                neg = True
        if mod and asp:
            return enumerate_pairs(mod, asp, neg)
    return []


def find_compound_aspect(t):
    # This enforces rules 8 and 9
    text = t.text.lower()
    # Find the entire compound phrase from a single token
    for child in t.children:
        if child.dep_ == 'compound':
            text = find_compound_aspect(child) + ' ' + text
    return text


def find_compound_modifier(t):
    # This enforces a new rule to fetch:
    # (10) advmod & npadvmod: very loud & user friendly
    # (11) xcomp phrases: able to connect
    text = t.text.lower()
    pre, post = '', ''
    for child in t.children:
        if child.dep_.endswith('advmod'):
            pre += child.text.lower() + ' '
        elif child.dep_ == 'xcomp':
            post = ' to ' + child.text.lower()
    return pre + text + post


def find_conjunctions(t):
    # This enforces rules 5 and 6
    conjunction_token = None
    has_valid_cc = False
    for child in t.children:
        if ((child.dep_ == 'punct' and child.text == ',') or \
            (child.dep_ == 'cc' and child.text.lower() == 'and')):
            has_valid_cc = True
        if child.dep_ == 'conj':
            conjunction_token = child
    if conjunction_token and has_valid_cc:
        return [t] + find_conjunctions(conjunction_token)
    return [t]


def enumerate_pairs(mod_token, asp_token, neg=False):
    pairs = []
    all_mods = find_conjunctions(mod_token)
    all_asps = find_conjunctions(asp_token)
    for mod in all_mods:
        mod_text = find_compound_modifier(mod)
        for asp in all_asps:
            asp_text = find_compound_aspect(asp)
            if neg:
                pairs.append(('not ' + mod_text, asp_text))
            else:
                pairs.append((mod_text, asp_text))
    return pairs


def main():
    args = parse_arguments()
    get_extractions_from_reviews(**vars(args))


if __name__ == "__main__":
    main()
