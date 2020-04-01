import os
import sys
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sampo import nn_report
from sampo import os_utils
from sklearn import preprocessing


def get_embedding_vector(ann, ext2idx, query):
    if query not in ext2idx:
        return None
    q_idx = ext2idx[query]
    q_vec = ann.get_item_vector(q_idx)
    q_vec_normalized = preprocessing.normalize([q_vec], norm='l2')
    return q_vec_normalized[0]


def get_embeddings(path, name, fact_path):
    exts = pd.read_csv(os.path.join(path, 'extr_index.csv'))
    ext2idx = dict(zip((x['modifier'] + ';' + x['aspect']
                        for _, x in exts.iterrows()), range(len(exts))))
    dim, embd_dim, iterations, name = os_utils.get_factorization_params_from_path(fact_path)
    embeddings = []
    indices = nn_report.load_nearest_neighbor_indices(fact_path)
    for i in range(iterations):
        report = pd.read_csv(os.path.join(fact_path, 'nn_report_' + str(i) + '.csv'))
        ann = indices[i]
        embd_map = {}
        query_cmap = {}
        for query, grp in report.groupby('query'):
            grp = grp.reset_index()
            query_color = len(query_cmap)
            query_cmap[query] = query_color
            nns = grp['neighbor'].tolist()
            q_embed = get_embedding_vector(ann, ext2idx, query)
            if q_embed is not None:
                embd_map[query] = (q_embed, query_color)
            for nn in nns:
                n_embed = get_embedding_vector(ann, ext2idx, nn)
                if n_embed is not None:
                    embd_map[nn] = (n_embed, query_color)

        embeddings.append(embd_map)
        # extractions = report['query'].drop_duplicates().tolist() + report['neighbor'].drop_duplicates().tolist()
        # extractions = list(set(extractions))
        #
        #
        # for q in extractions:
        #     if q not in ext2idx:
        #         continue
        #     q_idx = ext2idx[q]
        #     q_vec = ann.get_item_vector(q_idx)
        #     q_vec_normalized = preprocessing.normalize([q_vec], norm='l2')
        #     embd_map[q] = q_vec_normalized[0]
        # embeddings.append(embd_map)
    return embeddings


def visualize(path, name='current'):
    fact_path = os_utils.get_factorization_path_by_name(name, path)
    embeddings = get_embeddings(path, name, fact_path)
    for iteration, embed_map in enumerate(embeddings):
        tsne_plot(embed_map, fact_path, iteration)


def tsne_plot(embed_map, fact_path, iteration):
    "Creates and TSNE model and plots it"
    labels = []
    embeds = []
    colors = []
    for key, value in embed_map.items():
        labels.append(key)
        embeds.append(value[0])
        colors.append(value[1])


    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(embeds)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    plt.scatter(x, y, c=colors)
    for i in range(len(x)):
        # plt.scatter(x[i], y[i], c=colors[i], cmap='virdis')
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(os.path.join(fact_path, "embd_viz_" + str(iteration) + ".png"))


if __name__ == '__main__':
    path = sys.argv[1]
    name = sys.argv[2]
    visualize(path, name)