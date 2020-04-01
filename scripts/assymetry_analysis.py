import pandas as pd
from collections import defaultdict
from annoy import AnnoyIndex
from tqdm import tqdm

def init():
    global indices
    indices = defaultdict(lambda: defaultdict(dict))
    for dim in ['matrix', 'tensor']:
        for size in [500, 5000]:#, 1000, 5000, 10000]:
            folder = 'data/' + dim + '/200x' + str(size)
            # loading the index
            t = AnnoyIndex(20, 'angular')
            t.load(folder + '/embd.ann')
            indices[dim][size]['index'] = t
            # loading the extractions
            exts = pd.read_csv(folder + '/extr_index.csv')
            ext2idx = dict(zip((x['modifier'] + ';' + x['aspect'] \
                                for _, x in exts.iterrows()), range(len(exts))))
            indices[dim][size]['exts'] = exts
            indices[dim][size]['ext2idx'] = ext2idx


def _match(str1, str2):
    str1 = str1[:-1] if str1[-1] == 's' else str1
    str2 = str2[:-1] if str2[-1] == 's' else str2
    if str1.startswith(str2) or str2.startswith(str1):
        return True
    if str1.endswith(str2) or str2.endswith(str1):
        return True
    return False


def get_symmetry_stats(queries, num_ext=5000, dim=200):
    for i, q in tqdm(queries.iterrows(), total=queries.shape[0]):
        all_nns = get_nn(q)[:10]
        for k, ans in enumerate(all_nns):
            ans_q = {"modifier": ans[0].split(" - ")[0], "aspect": ans[0].split(" - ")[1]}
            query_rank = get_rank(q, ans_q)
            query_str = q['modifier'] + " - " + q["aspect"]
            print("query: {}, neighbor: {}, sim: {}, rank: {}, reverse_rank: {}".format(query_str, ans[0], ans[1], k, query_rank))


def get_rank(query, ans):
    query_str = query['modifier'] + " - " + query["aspect"]
    ans_neighbors = get_nn(ans)
    for i, nn in enumerate(ans_neighbors):
        if query_str == nn[0]:
            return i
    return -1


def get_nn(query, num_ext=5000, dim=200):
    mod, asp = query['modifier'], query['aspect']
    extractions = indices['tensor'][num_ext]['exts']
    ext2idx = indices['tensor'][num_ext]['ext2idx']
    try:
        ext_idx = ext2idx[mod + ';' + asp]
    except KeyError:
        return 'The input is not found in the KB'
    # fetching the index
    t = indices['tensor'][num_ext]['index']
    results, dists = t.get_nns_by_item(ext_idx, 100, include_distances=True)
    if ext_idx in results:
        remove_idx = results.index(ext_idx)
        results.pop(remove_idx)
        dists.pop(remove_idx)
    # filtering out unwanted results
    final_results, final_dists = [], []
    for (r, d) in zip(results, dists):

        nn_mod = extractions.iloc[r]['modifier']
        nn_asp = extractions.iloc[r]['aspect']
        include_neighbor = not _match(nn_asp, asp)
        if include_neighbor:
            final_results.append(nn_mod + ' - ' + nn_asp)
            final_dists.append(1 - d / 2)
    return list(zip(final_results, final_dists))


def main():
    init()
    queries = pd.read_csv('query.csv')
    get_symmetry_stats(queries)


if __name__ == '__main__':
    main()
