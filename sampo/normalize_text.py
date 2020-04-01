import os
import spacy
import argparse
import pandas as pd
from tqdm import tqdm
from spacy.matcher import Matcher

from sampo.os_utils import create_folder_if_absent


tqdm.pandas()  # adding progress bar for operations on dataframes


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='The csv file storing extractions (required \
                        columns: item_id, review_id, modifier, aspect)')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The folder in which normalized extractions and \
                        their mappings will be stored.')
    args = parser.parse_args()
    return args


def load_tools():
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    # building a matcher for matching aspects
    aspect_matcher = Matcher(nlp.vocab)
    adv_pattern = [{'TAG': {'IN': ['RB']},'DEP': {'NOT_IN': ['neg']}},
                   {'TAG': {'IN': ['JJ', 'RB', 'NNP']}}]
    aspect_matcher.add("adv", None, adv_pattern)
    # building a matcher for matching mofidiers
    modifier_matcher = Matcher(nlp.vocab)
    adv_pattern = [{'TAG': {'IN': ['RB']},'DEP': {'NOT_IN': ['neg']}},
               {'TAG': {'IN': ['JJ', 'RB', 'NNP']}}]
    prep_pattern = [{'TAG': {'IN': ['IN']}},
               {'TAG': {'IN': ['NN']}}]
    modifier_matcher.add("adv", None, adv_pattern)
    modifier_matcher.add("prep", None, prep_pattern)
    return nlp, aspect_matcher, modifier_matcher


def normalize_text(text_input, matcher, nlp):
    normalized_text = drop_adverb_and_lemmatize(text_input, matcher, nlp)
    return normalized_text


def drop_adverb_and_lemmatize(text, matcher, nlp):
    doc = nlp(str(text), disable=["ner", "parser"])
    # drop determiners
    if len(doc) > 1:
        trimmed_text = ' '.join(t.text for t in doc if t.tag_ not in ['DT', 'UH'])
        doc = nlp(trimmed_text)
    matches = matcher(doc)
    if len(matches) == 0:
        return ' '.join([t.lemma_ for t in doc])
    match_id, start, end = matches[0]
    span = doc[(start + 1):end]
    normalized = ' '.join([doc[:start].lemma_, span.lemma_, doc[end:].lemma_])
    return normalized.strip()


def get_pos_pattern(text, nlp):
    return [t.tag_ for t in nlp(str(text), disable=["ner", "parser"])]


def normalize_modifers(data, path, nlp, modifier_matcher):
    mod_df = data['modifier'].drop_duplicates().to_frame()
    print('Normalizing modifiers ...')
    mod_df['norm_mod'] = mod_df['modifier'].progress_apply(
        lambda x: normalize_text(x, modifier_matcher, nlp))
    mod_df.to_csv(os.path.join(path, 'modifier_mapping.csv'), index=None)
    data = pd.merge(data, mod_df, on='modifier', how='left')
    print('Done')
    return data


def normalize_aspects(data, path, nlp, aspect_matcher):
    asp_df = data['aspect'].drop_duplicates().to_frame()
    print('Normalizing aspects ...')
    asp_df['norm_asp'] = asp_df['aspect'].progress_apply(
        lambda x: normalize_text(x, aspect_matcher, nlp))
    asp_df.to_csv(os.path.join(path, 'aspect_mapping.csv'), index=None)
    data = pd.merge(data, asp_df, on='aspect', how='left')
    print('Done')
    return data


def normalize_data(filename, path):
    create_folder_if_absent(path)
    data = pd.read_csv(filename)
    nlp, aspect_matcher, modifier_matcher = load_tools()  # loading nlp tools
    data = normalize_modifers(data, path, nlp, modifier_matcher)
    data = normalize_aspects(data, path, nlp, aspect_matcher)
    # storing the results
    data = data[['item_id', 'review_id', 'norm_mod', 'norm_asp']]
    data = data.rename(columns={"norm_mod": "modifier", "norm_asp": "aspect"})
    data.to_csv(os.path.join(path, 'merged_data.csv'), index=None)


def main():
    args = parse_arguments()
    normalize_data(**vars(args))


if __name__ == '__main__':
    main()
