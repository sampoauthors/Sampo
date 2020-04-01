import math
import argparse
import numpy as np
import pandas as pd
from os.path import join
from tabulate import tabulate
from numpy.linalg import norm

from sampo.os_utils import get_all_factorization_names
from sampo.os_utils import get_factorization_path_by_name
from sampo.nn_report import load_nearest_neighbor_indices
from sampo.os_utils import get_factorization_params_from_path
from sampo.nn_report import find_nearest_neighbors, aggregate_report


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path to the folder containing the tensor or \
                        the matrix.')
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='The csv file containing the ground-truth -- \
                        columns "modifier", "aspect", "other_modifier", \
                        "other_aspect" and "label" should be present.')
    parser.add_argument('-n', '--neighbors', type=int, default=10,
                        help='The number of neighbors to return and evaluate.')
    parser.add_argument('-c', '--cutoff', type=int, default=200,
                        help='The number of neighbors fetched from the system \
                        before filtering.')
    parser.add_argument('--names', default=['current'], nargs='+',
                        help='The names of factorizations used in evaluation.')
    parser.add_argument('--relaxed', action='store_true',
                        help='Ignore edge direction in evaluation.')
    parser.add_argument('--sort_by', type=str, default='sim',
                        help='the method used to rank aggregated results.')
    args = parser.parse_args()
    return args


def eval_report_nmap(report, atk, pair2label):
    nmaps = 0
    for q in report:
        nns = report[q]['nns']
        if len(nns) < atk:  # not enough to evaluate
            q_map = np.NaN
        else:
            q_map, correct_count = 0, 0
            for pos in range(atk):
                if pair2label[q + ' ' + nns[pos]['text']] == 1:
                    correct_count += 1
                    q_map += (correct_count / (pos + 1))
            q_map = q_map / atk
        nmaps += q_map
    return (nmaps / len(report))


def eval_report_incomplete_nmap(report, atk, pair2label, pair2junk):
    nmaps = 0
    valid_report = 0
    for q in report:
        nns = report[q]['nns']
        q_map, correct_count = 0, 0
        valid_count = 0
        for pos in range(min(atk, len(nns))):
            if pair2junk[q + ' ' + nns[pos]['text']] == 1:
                continue
            if pair2label[q + ' ' + nns[pos]['text']] == 1:
                correct_count += 1
                q_map += (correct_count / (valid_count + 1))
            valid_count += 1
        if valid_count == 0:
            continue
        valid_report += 1
        q_map = q_map / valid_count
        nmaps += q_map
    return (nmaps / valid_report)


def eval_universal_map(all_sims):
    univ_map, correct_count = 0, 0
    for pos, (sim, label) in enumerate(all_sims):
        if label == 1:
            correct_count += 1
            univ_map += (correct_count / (pos + 1))
    univ_map = univ_map / len(all_sims)
    return univ_map


def eval_universal_precision(all_sims):
    correct_count = 0
    for pos, (sim, label) in enumerate(all_sims):
        if label == 1:
            correct_count += 1
    return correct_count / len(all_sims)


def eval_report_ndcg(report, atk, pair2label):
    ndcgs = 0
    for q in report:
        nns = report[q]['nns']
        if len(nns) < atk:  # not enough to evaluate
            q_dcg = np.NaN
        else:
            q_dcg = 0
            for pos in range(atk):
                if pair2label[q + ' ' + nns[pos]['text']] == 1:
                    q_dcg += (1.0 / math.log2(pos + 2))
        ndcgs += q_dcg
    return (ndcgs / len(report))


def clean_queries(queries, neighbors):
    # Dropping any query where number of available labels < neighbors
    counts = queries.groupby(['modifier', 'aspect']).count().reset_index()
    # keeping frequent queries
    counts = counts[counts['label'] >= neighbors]
    # keeping the frequent subset
    queries = pd.merge(queries, counts[['modifier', 'aspect']],
                       on=['modifier', 'aspect'], how='inner')
    return queries


def get_pair_dist(pair, exts, ext2idx, indices):
    sideA = pair['modifier'] + ';' + pair['aspect']
    sideB = pair['other_modifier'] + ';' + pair['other_aspect']
    if sideA not in ext2idx or sideB not in ext2idx:
        return None
    A_idx, B_idx = ext2idx[sideA], ext2idx[sideB]
    sim = 0.0
    for i, ann in enumerate(indices):
        A_vec = ann.get_item_vector(A_idx)
        B_vec = ann.get_item_vector(B_idx)
        sim += np.dot(A_vec, B_vec) / (norm(A_vec) * norm(B_vec))
    return sim / len(indices)


def run_evaluation(path, filename, neighbors, cutoff, names, sort_by,
                   relaxed=False):
    # checking if the arguments are valid (and set default values)
    assert sort_by in ['rank', 'sim'], "Invalid sort_by argument."
    # adding all factorization paths that need to be considered
    all_fact_paths = []
    if 'all' in names:
        names = get_all_factorization_names(path)
    for name in names:
        curr_path = get_factorization_path_by_name(name, path)
        if curr_path:
            all_fact_paths.append(curr_path)
        else:
            print('Factorization named ' + name + ' does not exist!')
    # loading the labeled examples
    queries = pd.read_csv(filename)
    queries = clean_queries(queries, neighbors)
    pair2label, labeled = {}, set()
    pair2junk = {}
    for _, row in queries.iterrows():
        key = ' '.join([row['modifier'] + ';' +  row['aspect'],
                        row['other_modifier'] + ';' + row['other_aspect']])
        if relaxed:
            pair2label[key] = 0 if row['rel_label'] == False else 1
        else:
            pair2label[key] = 0 if row['label'] == False else 1
        pair2junk[key] = 0 if row['junk_row'] == False else 1
    labeled = list(pair2label.keys())
    table = []
    for p in all_fact_paths:
        indices = load_nearest_neighbor_indices(p)
        dim, embd_dim, iterations, name = get_factorization_params_from_path(p)
        row = [name, dim, embd_dim, iterations]
        # running the report
        report = find_nearest_neighbors(filename, path, indices, cutoff, False,
                                        False, False)
        agg_report = aggregate_report(report, indices, cutoff, sort_by,
                                      neighbors, labeled)
        all_k = list(range(0, neighbors, 5))[1:]
        all_k = all_k if neighbors in all_k else all_k + [neighbors]
        # evaluating MAP & NDCG scores
        for k in all_k:
            row.append(eval_report_nmap(agg_report, k, pair2label))
        for k in all_k:
            row.append(eval_report_incomplete_nmap(agg_report, k, pair2label, pair2junk))
        #for k in all_k:
        #    row.append(eval_report_ndcg(agg_report, k, pair2label))
        # Ranking all labeled pairs for Universal Rankings
        exts = pd.read_csv(join(path, 'extr_index.csv'))
        ext2idx = dict(zip((x['modifier'] + ';' + x['aspect']
                            for _, x in exts.iterrows()), range(len(exts))))
        all_sims = []
        for _, pa in queries.iterrows():
            sim = get_pair_dist(pa, exts, ext2idx, indices)
            if sim is None:
                continue
            key = ' '.join([pa['modifier'] + ';' + pa['aspect'],
                            pa['other_modifier'] + ';' + pa['other_aspect']])
            label = pair2label[key]
            junk = pair2junk[key]
            if junk == 1:
                continue
            all_sims.append((sim, label))
        all_sims = sorted(all_sims, reverse=True)
        all_p = [.10, .20, .50]
        #for perc in all_p:
        #    row.append(eval_universal_map(all_sims[:int(len(all_sims) * perc)]))
        # Ranking all labeled pairs in the neighborhood
        for k in all_k:
            all_neighborhood_sims = []
            for query in agg_report:
                for itt in range(k):
                    if itt >= len(agg_report[query]['nns']):
                        continue
                    neighbor = agg_report[query]['nns'][itt]['text']
                    dictrow = {}
                    dictrow['modifier'] = query.split(';')[0]
                    dictrow['aspect'] = query.split(';')[1]
                    dictrow['other_modifier'] = neighbor.split(';')[0]
                    dictrow['other_aspect'] = neighbor.split(';')[1]
                    key = ' '.join([dictrow['modifier'] + ';' +  dictrow['aspect'],
                                    dictrow['other_modifier'] + ';' + dictrow['other_aspect']])
                    label = pair2label[key]
                    junk = pair2junk[key]
                    if junk == 1:
                        continue
                    sim = get_pair_dist(dictrow, exts, ext2idx, indices)
                    all_neighborhood_sims.append((sim, label))
            all_neighborhood_sims = sorted(all_neighborhood_sims, reverse=True)
            for perc in all_p:
                row.append(eval_universal_precision(
                    all_neighborhood_sims[:int(len(all_neighborhood_sims) * perc)]))
        table.append(row)

    table = list(sorted(table))
    print(tabulate(table, headers=['Name', 'Type', 'Dim.', 'Trials'] +
                   ['MAP@' + str(x) for x in all_k] +
                   ['PMAP@' + str(x) for x in all_k] +
                   #['NDCG@' + str(x) for x in all_k] +
                   #['UMAP' + str(int(x * 100)) + '%' for x in all_p] +
                   ['PREC@N' + str(x) + '-' + str(int(y * 100)) + '%'
                    for x in all_k for y in all_p],
                   tablefmt='orgtbl',  floatfmt=".3f"))


def main():
    args = parse_arguments()
    run_evaluation(**vars(args))


if __name__ == "__main__":
    main()
