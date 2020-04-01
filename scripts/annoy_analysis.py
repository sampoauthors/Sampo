import random
import argparse
from tqdm import tqdm
from annoy import AnnoyIndex


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annoy_file', type=str, required=True,
                        help='The path to the annoy index file.')
    parser.add_argument('-e', '--extractions', type=int, required=True,
                        help='Number of extractions index by annoy.')
    parser.add_argument('-d', '--dimension', type=int, required=True,
                        help='The embedding dimension.')
    parser.add_argument('-q', '--queries', type=int, default=20,
                        help='The number of queries to be tested.')
    parser.add_argument('-t', '--topk', type=int, default=20,
                        help='The number of top neighbors to compare.')
    parser.add_argument('-r', '--repetitions', type=int, default=10,
                        help='The number of repetition in the experiment.')
    args = parser.parse_args()
    return args


def measure_stability(annoy_file, extractions, dimension,
                      queries=20, topk=20, repetitions=10):
    # loading the index
    t = AnnoyIndex(dimension, 'angular')
    t.load(annoy_file)
    # reading all vectors
    vecs = [t.get_item_vector(i) for i in range(extractions)]
    # sampling quries
    q_inds = random.sample(list(range(extractions)), queries)
    # repeating the process of building indices
    all_indices = [t]
    for _ in tqdm(range(repetitions - 1)):
        t = AnnoyIndex(dimension, 'angular')
        for i, v in enumerate(vecs):
            t.add_item(i, v)
        t.build(100)
        all_indices.append(t)
    # checking if the results are the same
    inconsistencies = 0
    for q in tqdm(q_inds):
        all_nns = set()
        for t in all_indices:
            all_nns.update(t.get_nns_by_item(q, topk))
        if len(all_nns) != topk:
            inconsistencies += 1
    # printing the results
    print('{} of the queries had inconsistent neighbors.'.format(inconsistencies
                                                                 / queries))


def main():
    args = parse_arguments()
    measure_stability(**vars(args))


if __name__ == "__main__":
    main()
