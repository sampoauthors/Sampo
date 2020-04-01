import argparse
import numpy as np
import pandas as pd
import json as json_module
from tqdm import tqdm
from os.path import join
from annoy import AnnoyIndex
from tabulate import tabulate
from numpy.linalg import norm
from collections import defaultdict

from sampo.os_utils import get_factorization_path_by_name
from sampo.os_utils import get_factorization_params_from_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path to the folder containing the tensor or \
                        the matrix.')
    parser.add_argument('-f', '--filename', type=str,
                        help='The csv file containing the extractions for \
                        which the nearest neighbor search is to be done -- \
                        columns "modifier" and "aspect" should be present.')
    parser.add_argument('-n', '--neighbors', type=int, default=10,
                        help='Number of neighbors to include in the report.')
    parser.add_argument('-c', '--cutoff', type=int, default=100,
                        help='The number of neighbors fetched from the system \
                        before filtering.')
    parser.add_argument('--name', default='current',
                        help='The name of factorization to be used to create \
                        the nearest neighbor report.')
    parser.add_argument('--unique_aspect', action='store_true',
                        help='Return NNs with different aspects')
    parser.add_argument('--unique_modifier', action='store_true',
                        help='Return NNs with different modifiers')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--json', type=str,
                       help='Path to save the output .json report')
    group.add_argument('--csv', type=str,
                       help='Path to save the output .csv report')
    parser.add_argument('--sort_by', type=str, default='rank',
                        help='the method use rank aggregated results.')
    parser.add_argument('-e', '--exact', action='store_true',
                        help='Return exact NNs (not approximations)')
    args = parser.parse_args()
    return args


def load_nearest_neighbor_indices(fact_path):
    indices = []
    dim, embd_dim, iterations, name = \
        get_factorization_params_from_path(fact_path)
    for i in range(iterations):
        t = AnnoyIndex(embd_dim, 'angular')
        t.load(join(fact_path, 'embd_' + str(i) + '.ann'))
        indices.append(t)
    return indices


def get_exact_nns_by_item(ann, num_vecs, q_vec, cutoff):
    sims = []
    for i in range(num_vecs):
        curr_vec = ann.get_item_vector(i)
        curr_sim = np.dot(curr_vec, q_vec) / (norm(curr_vec) * norm(q_vec))
        sims.append(curr_sim)
    # sort and return the cutoff
    results = list(sorted([(v, i) for i, v in enumerate(sims)], reverse=True))
    results = [list(t) for t in zip(*results[:cutoff])]
    return results[1], results[0]


def get_query_report(q, exts, ext2idx, indices, cutoff, unique_modifier,
                     unique_aspect, exact):
    # checking if the query exist among the extractions (and getting the index)
    query = q['modifier'] + ';' + q['aspect']
    if query not in ext2idx:
        print('Query "' + query + '"" not found in the data!')
        return None
    q_idx = ext2idx[query]
    # getting query count if available
    q_count = int(q['count']) if 'count' in q else None
    # making the query report
    q_report = {}
    q_report['nns'] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    q_report['count'] = q_count
    q_report['lowest_sim'] = []
    # iterating over all approximate nearest neighbor indices
    for i, ann in enumerate(indices):
        q_vec = ann.get_item_vector(q_idx)
        if exact:
            nns, dists = get_exact_nns_by_item(ann, len(exts), q_vec, cutoff)
        else:
            nns, dists = \
                ann.get_nns_by_item(q_idx, cutoff, include_distances=True)
        # removing the query from the results
        if q_idx in nns:
            remove_idx = nns.index(q_idx)
            nns.pop(remove_idx)
            dists.pop(remove_idx)
        # going over the results and only reporting the desired ones
        curr_rank = 0
        sim = np.NaN
        for (nn_idx, d) in zip(nns, dists):
            nn_mod = exts.iloc[nn_idx]['modifier']
            nn_asp = exts.iloc[nn_idx]['aspect']
            nn = nn_mod + ';' + nn_asp
            # applying filters!
            if unique_modifier and _match(nn_mod, q['modifier']):
                continue
            if unique_aspect and _match(nn_asp, q['aspect']):
                continue
            # add the nearest neighbor to the report
            nn_vec = ann.get_item_vector(nn_idx)
            sim = np.dot(nn_vec, q_vec) / (norm(nn_vec) * norm(q_vec))
            q_report['nns'][nn]['rank'][i] = curr_rank
            q_report['nns'][nn]['sim'][i] = float(sim)
            q_report['nns'][nn]['count'] = int(exts.iloc[nn_idx]['count'])
            q_report['nns'][nn]['text'] = nn
            curr_rank += 1
        q_report['lowest_sim'].append(sim)
    return q_report


def find_nearest_neighbors(filename, path, indices, cutoff, unique_modifier,
                           unique_aspect, exact):
    queries = pd.read_csv(filename).drop_duplicates(
        subset=['modifier', 'aspect'], keep='last')
    # reading the extractions & creating an inverted index
    exts = pd.read_csv(join(path, 'extr_index.csv'))
    ext2idx = dict(zip((x['modifier'] + ';' + x['aspect']
                        for _, x in exts.iterrows()), range(len(exts))))
    # iterating over queries and creating a report
    report = {}   # keys are queries and values are individual reports
    for _, q in tqdm(queries.iterrows(), total=len(queries)):
        query = q['modifier'] + ';' + q['aspect']
        q_report = get_query_report(q, exts, ext2idx, indices, cutoff,
                                    unique_modifier, unique_aspect, exact)
        if q_report is None:
            continue
        report[query] = q_report
    return report


def _match(str1, str2):
    str1 = str1[:-1] if str1[-1] == 's' else str1
    str2 = str2[:-1] if str2[-1] == 's' else str2
    if str1.startswith(str2) or str2.startswith(str1):
        return True
    if str1.endswith(str2) or str2.endswith(str1):
        return True
    return False


def aggregate_report(report, indices, cutoff, sort_by, neighbors, labeled=None):
    trials = len(indices)
    for query, q_report in report.items():
        for nn in q_report['nns']:
            # aggregating similarity
            sim_list = []
            for i in range(trials):
                if i in q_report['nns'][nn]['sim']:
                    sim_list.append(q_report['nns'][nn]['sim'][i])
                else:
                    sim_list.append(q_report['lowest_sim'][i])
            q_report['nns'][nn]['sim'] = float(np.mean(sim_list))
            q_report['nns'][nn]['sim_std'] = float(np.std(sim_list))
            q_report['nns'][nn]['sim_min_max'] = (min(sim_list), max(sim_list))
            # aggreating ranks
            rank_list = []
            for i in range(trials):
                if i in q_report['nns'][nn]['rank']:
                    rank_list.append(q_report['nns'][nn]['rank'][i])
                else:
                    rank_list.append(cutoff)
            q_report['nns'][nn]['rank'] = float(np.mean(rank_list))
            q_report['nns'][nn]['rank_std'] = float(np.std(rank_list))
            q_report['nns'][nn]['rank_min_max'] = (min(rank_list), max(rank_list))
    # sort and keep the top results
    return sort_and_crop_report(report, sort_by, neighbors, labeled)


def sort_and_crop_report(report, sort_by, neighbors, labeled):
    for query in report:
        nns = report[query]['nns']
        if sort_by == 'rank':
            sorted_nn = list(sorted(nns.values(), key=lambda x: x['rank']))
        else:
            sorted_nn = list(sorted(nns.values(), key=lambda x: x['sim'],
                                    reverse=True))
        sorted_nn = sorted_nn
        report[query]['nns'] = []
        found = 0
        for nn in sorted_nn:
            if labeled and query + ' ' + nn['text'] not in labeled:
                continue
            report[query]['nns'].append(nn)
            found += 1
            if found == neighbors:
                break
    return report


def output_report(report, csv, json):
    if json:
        with open(json, 'w') as jsonfile:
            json_module.dump(report, jsonfile, indent=2)
    elif csv:
        all_values = []
        for q, q_report in report.items():
            all_values += [[q, q_report['count'], x['text'], x['count'],
                            x['sim'], x['sim_std'], x['sim_min_max'], x['rank'], x['rank_std'], x['rank_min_max']] for x in q_report['nns']]
        dat = pd.DataFrame(all_values, columns=['query', 'q_count', 'neighbor',
                                                'n_count', 'sim', 'sim_std', 'sim_min_max', 'rank', 'rank_std', 'rank_min_max'])
        dat.to_csv(csv, index=False)
    else:
        for q, q_report in report.items():
            print('Query: {}, Count: {}'.format(q, q_report['count']))
            table = [[x['text'], x['count'], x['sim'],
                      x['rank']] for x in q_report['nns']]
            print(tabulate(table,
                           headers=['nn', 'count', 'sim', 'sim_std', 'sim_min_max', 'rank', 'rank_std', 'rank_min_max'],
                           tablefmt='orgtbl'))
            print('=' * 20)


def run_report(path, filename=None, neighbors=10, cutoff=100, name='current',
               unique_aspect=False, unique_modifier=False, sort_by='rank',
               csv=None, json=None, exact=False):
    # checking if the arguments are valid (and set default values)
    if filename is None:
        filename = join(path, 'extr_index.csv')
    fact_path = get_factorization_path_by_name(name, path)
    assert fact_path is not None, "Invalid path or name."
    assert sort_by in ['rank', 'sim'], "Invalid sort_by argument."
    # loading necessary indices and running the report
    indices = load_nearest_neighbor_indices(fact_path)
    report = find_nearest_neighbors(filename, path, indices, cutoff,
                                    unique_modifier, unique_aspect, exact)
    agg_report = aggregate_report(report, indices, cutoff, sort_by, neighbors)
    output_report(agg_report, csv, json)


def main():
    args = parse_arguments()
    run_report(**vars(args))


if __name__ == "__main__":
    main()
