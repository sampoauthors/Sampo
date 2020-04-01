import pandas as pd
import argparse
import numpy as np
import os
from tqdm import tqdm
from sampo import parafac
from sampo import nn_report
from sampo import os_utils
from tabulate import tabulate

from sampo.os_utils import get_factorization_path_by_name


def sample_queries(src, bin_count=5, queries_per_bin=5):
    extractions_df = pd.read_csv(src)

    extractions_df['bin'] = pd.cut(extractions_df['count'], bin_count)

    sampled_queries = extractions_df.groupby('bin').apply(lambda x: x.sample(n=min(queries_per_bin, len(x)))).reset_index(drop=True)
    sampled_queries = sampled_queries[['modifier', 'aspect', 'count', 'bin']]
    return sampled_queries


def evaluate_noise_stability(path, factorization_prefix, queries_sample_src, dimension, iterations=5):
    factorization_name = factorization_prefix + "Noise" + str(dimension)
    noise = 'genCPU' + str(dimension)
    parafac.run_parafac(path, iterations=iterations, is_matrix=True, name=factorization_name, noise=noise)
    mean_std, std_std, rank_mean_std, rank_stability = evaluate_factorization_stability(path, factorization_name, queries_sample_src)
    stability = evaluate_stability_percentage(path, factorization_name, queries_sample_src)
    print("{}\t{}\t{}\t{}\n".format(mean_std, std_std, rank_mean_std, stability))
    return mean_std, std_std, rank_mean_std, stability


def evaluate_factorization_stability(path, factorization_name, queries_sample_src):
    report_dest = os.path.join(os_utils.get_factorization_path_by_name(factorization_name, path), "stability_iterations.csv")
    nn_report.run_report(path, queries_sample_src, csv=report_dest, name=factorization_name)
    report = pd.read_csv(report_dest)
    queries_sample = pd.read_csv(queries_sample_src)
    queries_sample['query'] = queries_sample.apply(lambda x: '{};{}'.format(x['modifier'], x['aspect']), axis=1)
    report = report.merge(queries_sample, on=['query'], how='left')
    report.to_csv(report_dest, index=None)
    mean_std = round(np.mean(report['sim_std']), 5)
    std_std = round(np.std(report['sim_std']), 5)
    rank_mean_std = round(np.mean(report['rank_std']), 5)
    stability = evaluate_stability_percentage(path, factorization_name, queries_sample_src)
    print("{}\t{}\t{}\t{}\n".format(mean_std, std_std, rank_mean_std, stability))
    return mean_std, std_std, rank_mean_std, stability


def evaluate_factorization_stability_per_embed_dim(path, factorization_prefix, queries_sample_src, max_embedding_dim=50, iterations=5):
    dims = list(range(10, max_embedding_dim, 10)) + [max_embedding_dim]
    stability = []
    for d in tqdm(dims):
        factorization_name = factorization_prefix + "Dim" + str(d)
        parafac.run_parafac(path, dimensions=d, iterations=iterations, is_matrix=True, name=factorization_name)
        mean_std, std_std, rank_mean_std, rank_stability = evaluate_factorization_stability(path, factorization_name, queries_sample_src)
        stability.append([d, mean_std, std_std, rank_mean_std, rank_stability])
    print(tabulate(stability, headers=['Dim', 'Sim_Mean_Std', 'Sim_Std_Std', 'Rank_Mean_Std', 'Rank Stability'], tablefmt='orgtbl'))
    out = "\t".join(["{} ({}) / {} / {}".format(t[1], t[2], t[3], t[4]) for t in stability])
    print(out)
    return stability


def evaluate_stability_percentage(path, factorization_name, queries_sample_src):
    fact_path = get_factorization_path_by_name(factorization_name, path)
    indices = nn_report.load_nearest_neighbor_indices(fact_path)
    nn_info = {}
    queries_sample = pd.read_csv(queries_sample_src)
    queries_sample['query'] = queries_sample.apply(lambda x: '{};{}'.format(x['modifier'], x['aspect']), axis=1)
    for index, t in enumerate(indices):
        factorization_path = get_factorization_path_by_name(factorization_name, path)
        csv = os.path.join(factorization_path, 'nn_report_{}.csv'.format(index))
        if not os.path.exists(csv):
            report = nn_report.find_nearest_neighbors(queries_sample_src, path, [t], 100, False, False, None, False)
            agg_report = nn_report.aggregate_report(report, [t], 100, 'rank', 10)
            nn_report.output_report(agg_report, csv, None)
        agg_report = pd.read_csv(csv)
        agg_report = agg_report.merge(queries_sample, on=['query'], how='left')
        nn_info[index] = agg_report
    iterations = len(indices)
    mean_stabilities = []
    for i in range(iterations):
        for j in range(i+1, iterations):
            df1 = nn_info[i]
            df2 = nn_info[j]
            stabilities = []
            for _, row in queries_sample.iterrows():
                query = row['query']
                df1subset = df1[df1['query'] == query]
                neighbors1 = []
                if len(df1subset) > 0:
                    neighbors1 = df1subset['neighbor'].tolist()
                df2subset = df2[df2['query'] == query]
                neighbors2 = []
                if len(df2subset) > 0:
                    neighbors2 = df2subset['neighbor'].tolist()
                stability = float(len(set(neighbors1).intersection(set(neighbors2)))) / float(max(len(set(neighbors1)), 10))
                stabilities.append(stability)
            mean_stability = float(np.mean(stabilities))
            mean_stabilities.append(mean_stability)
    overall_mean_stability = float(np.mean(mean_stabilities))
    # print(overall_mean_stability)
    return overall_mean_stability



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path to the folder that contains the \
                            tensor/matrix to be factorized')
    parser.add_argument('-q', '--queries', type=str,
                        help='The queries to sample from')
    parser.add_argument('-s', '--sampled_queries', type=str, required=True,
                        help='The sampled queries path')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='The name of the factorization')
    parser.add_argument('-d', '--dim', type=int, default=50,
                        help='Max embedding dimension')
    parser.add_argument('-i', '--iterations', type=int, default=5,
                        help='Num of iterations')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--iter', action='store_true',
                       help='Evaluate stability of factorization')
    group.add_argument('--embed', action='store_true',
                       help='Evaluate stability of factorization per embedding dim')
    group.add_argument('--noise', action='store_true',
                       help='Evaluate stability per noise')

    args = parser.parse_args()
    if not os.path.exists(args.sampled_queries):
        queries = sample_queries(args.queries)
        queries.to_csv(args.sampled_queries, index=None)
    if args.iter:
        evaluate_factorization_stability(args.path, args.name, args.sampled_queries)
    if args.embed:
        evaluate_factorization_stability_per_embed_dim(args.path, args.name, args.sampled_queries, max_embedding_dim=args.dim, iterations=args.iterations)
    elif args.noise:
        evaluate_noise_stability(args.path, args.name, args.sampled_queries, args.dim, iterations=args.iterations)
