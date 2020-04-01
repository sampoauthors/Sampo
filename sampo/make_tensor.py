import os
import json
import sparse
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sampo.os_utils import create_folder_if_absent


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='The csv file storing the extractions (columns \
                        should be: item_id, review_id, modifier, aspect).')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path to the output directory.')
    parser.add_argument('-e', '--num_extractions', type=int, default=500,
                        help='Number of (most frequent) extractions used for \
                        creating the tensors.')
    parser.add_argument('-i', '--num_items', type=int, default=100,
                        help='Number of (most reviewed) items used for \
                        creating the tensors.')
    args = parser.parse_args()
    return args


def get_frequent_items(data, items):
    item_freq = data.groupby('item_id').size().to_frame('count').reset_index()
    items = min(items, len(item_freq))
    sorted_items = item_freq.sort_values('count')[-items:][['item_id']]
    data = pd.merge(data, sorted_items, on='item_id')
    return data


def get_frequent_extractions(data, extractions):
    ext_freq = data.groupby(['modifier',
                             'aspect']).size().to_frame('count').reset_index()
    extractions = min(extractions, len(ext_freq))
    sorted_exts = ext_freq.sort_values('count')[-extractions:][['modifier',
                                                                'aspect']]
    data = pd.merge(data, sorted_exts, on=['modifier', 'aspect'])
    return data


def slice_data(data, num_items, num_extractions):
    data = get_frequent_items(data, num_items)
    data = get_frequent_extractions(data, num_extractions)
    return data


def build_matrix(data, path):
    # getting all items (and building an inverted index)
    items = list(data['item_id'].unique())
    item2idx = dict(zip(items, range(len(items))))
    # getting all extractions (and building an inverted index)
    extractions = data.groupby(['modifier', 'aspect']).size().\
            to_frame('count').reset_index()
    ext2idx = dict(zip((x['modifier'] + ';' + x['aspect'] for _, x in
                        extractions.iterrows()), range(len(extractions))))
    # building inverted indices for modifier and aspects as well
    modifiers = list(data['modifier'].unique())
    aspects = list(data['aspect'].unique())
    # building the sparse matrix from the data
    values, x_idx, y_idx = [], [], []
    grouped_data = data.groupby(['item_id', 'modifier', 'aspect']).size().\
            to_frame('count').reset_index()
    for _, row in tqdm(grouped_data.iterrows(), total=len(grouped_data)):
        x_idx.append(item2idx[row['item_id']])
        y_idx.append(ext2idx[row['modifier'] + ';' + row['aspect']])
        values.append(row['count'])
    matrix = sparse.COO(np.array([x_idx, y_idx], dtype=np.int32),
                        np.array(values, dtype=np.float64),
                        shape=(len(items), len(extractions)))
    print("Sparsity: %f" % (len(values) / (len(items) * len(extractions))))
    # saving the results
    save_ndarray(matrix, items, modifiers, aspects, extractions, path)


def build_tensor(data, path):
    # getting all items (and building an inverted index)
    items = list(data['item_id'].unique())
    item2idx = dict(zip(items, range(len(items))))

    # getting all extractions (and building an inverted index)
    extractions = data.groupby(['modifier', 'aspect']).size().\
            to_frame('count').reset_index()

    # building inverted indices for modifier and aspects as well
    modifiers = list(data['modifier'].unique())
    mod2idx = dict(zip(modifiers, range(len(modifiers))))
    aspects = list(data['aspect'].unique())
    asp2idx = dict(zip(aspects, range(len(aspects))))

    # building the sparse matrix from the data
    values, x_idx, y_idx, z_idx = [], [], [], []
    grouped_data = data.groupby(['item_id', 'modifier', 'aspect']).size().\
            to_frame('count').reset_index()
    for _, row in tqdm(grouped_data.iterrows(), total=len(grouped_data)):
        x_idx.append(item2idx[row['item_id']])
        y_idx.append(mod2idx[row['modifier']])
        z_idx.append(asp2idx[row['aspect']])
        values.append(row['count'])
    tensor = sparse.COO(np.array([x_idx, y_idx, z_idx], dtype=np.int32),
                        np.array(values, dtype=np.float64),
                        shape=(len(items), len(modifiers), len(aspects)))
    print("Sparsity: %f" % (len(values) /
                            (len(items) * len(modifiers) * len(aspects))))
    # saving the results
    save_ndarray(tensor, items, modifiers, aspects, extractions, path)


def save_ndarray(tensor, items, modifiers, aspects, extractions, path):
    extractions.to_csv(os.path.join(path, "extr_index.csv"), index=False)
    if len(tensor.shape) == 2:  # if matrix
        sparse.save_npz(os.path.join(path, "matrix.npz"), tensor)
    else:
        sparse.save_npz(os.path.join(path, "tensor.npz"), tensor)
    # saving the meta-data
    meta_data = {'items': len(items), 'extractions': len(extractions),
                 'item_index': items, 'mod_index': modifiers,
                 'asp_index': aspects}
    with open(os.path.join(path, "meta_data.json"), 'w') as outfile:
        json.dump(meta_data, outfile)


def build_all_ndarrays(filename, path, num_items=100, num_extractions=500):
    create_folder_if_absent(path)
    # reading the data and getting the desired subset
    full_data = pd.read_csv(filename, dtype={'item_id': object})
    data = slice_data(full_data, num_items, num_extractions)
    # building both the matrix and the tensor
    build_matrix(data, path)
    build_tensor(data, path)


def main():
    args = parse_arguments()
    build_all_ndarrays(**vars(args))


if __name__ == "__main__":
    main()
