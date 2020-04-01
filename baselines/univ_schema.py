import time
import json
import torch
import random
import sparse
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from os.path import join
from annoy import AnnoyIndex
from torch.utils.data import DataLoader, Dataset

from sampo.os_utils import is_ndarray_folder
from sampo.os_utils import create_folder_if_absent, delete_factorization_by_name


class UniversalSchemaModel(nn.Module):
    def __init__(self, num_items, num_exts, embedding_dim=20):
        super(UniversalSchemaModel, self).__init__()
        # Creating embeddings for items and extractions
        self.I = nn.Embedding(num_items, embedding_dim)
        self.E = nn.Embedding(num_exts, embedding_dim)

    def forward(self, batch):
        item_embds = self.I(batch[:,0])
        ext_embds = self.E(batch[:,1])
        dot_prod = torch.bmm(item_embds.unsqueeze(1),
                             ext_embds.unsqueeze(2)).squeeze()
        return dot_prod


class UniversalSchema3DModel(nn.Module):
    def __init__(self, num_items, num_mods, num_asps, embedding_dim=20):
        super(UniversalSchema3DModel, self).__init__()
        # Creating embeddings for items, modifiers and aspects
        self.I = nn.Embedding(num_items, embedding_dim)
        self.M = nn.Embedding(num_mods, embedding_dim)
        self.A = nn.Embedding(num_asps, embedding_dim)

    def forward(self, batch):
        item_embds = self.I(batch[:,0])
        mod_embds = self.M(batch[:,1])
        asp_embds = self.A(batch[:,2])
        ext_embds = torch.mul(mod_embds.unsqueeze(2), asp_embds.unsqueeze(2))
        dot_prod = torch.bmm(item_embds.unsqueeze(1), ext_embds).squeeze()
        return dot_prod


class NDArrayDataset(Dataset):
    def __init__(self, path, neg_sample_ratio, min_freq, is_matrix=True):
        ndarray_file = 'matrix.npz' if is_matrix else 'tensor.npz'
        self.ndarray = sparse.load_npz(join(path, ndarray_file)).todense()
        # Creating positive instances
        nonzeros = (self.ndarray > min_freq).nonzero()
        print('# of non-zero cells is {}'.format(len(nonzeros[0])))
        print('Density is {}%'.format(len(nonzeros[0]) / self.ndarray.size))
        pos_points = list(zip(*nonzeros))
        # Creating negative instances (if there are enough negatives)
        assert len(nonzeros[0]) * (neg_sample_ratio + 1) < self.ndarray.size
        neg_points, seen_points = [], set(pos_points)
        while len(neg_points) < len(nonzeros[0]) * neg_sample_ratio:
            sample = tuple([np.random.choice(d) for d in self.ndarray.shape])
            if sample not in seen_points:
                seen_points.add(tuple(sample))
                neg_points.append(tuple(sample))
        # Combining all points and labels
        self.points = [np.array(x) for x in (pos_points + neg_points)]
        self.labels = [1] * len(pos_points) + [0] * len(neg_points)
        # Shuffling the points
        zipped = list(zip(self.points, self.labels))
        random.shuffle(zipped)
        self.points, self.labels = zip(*zipped)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path to the folder that contains the \
                        tensor/matrix to be factorized')
    parser.add_argument('-d', '--dimensions', type=int, default=20,
                        help='The desired embedding dimension (i.e., rank)')
    parser.add_argument('--name', default='current' ,
                        help='The name associated with the factorization \
                        results. This name can be used later for evalution \
                        and other analysis on the results.')
    parser.add_argument('--neg_ratio', type=int, default=2,
                        help='Number of neg samples per each pos sample')
    parser.add_argument('--min_freq', type=int, default=20,
                        help='Min Frequency to fill a cell with a one when' +
                        'when converting the counts into binary values.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tensor', '--3d', action='store_true',
                       help='Factorize only the tensor.')
    group.add_argument('--matrix', '--2d', action='store_true',
                       help='Factorize only the matrix.')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    args.is_matrix = args.matrix
    del args.matrix
    del args.tensor
    return args



def run_universal_schema(path, dimensions=20, name='current', neg_ratio=2,
                         min_freq=20, is_matrix=True, max_epochs=200,
                         batch_size=50, learning_rate=.001):
    # checking if the arguments are valid
    assert is_ndarray_folder(path), "Invalid path."
    assert '_' not in name, "Name cannot contain the '_' symbol."
    # loading the meta data
    with open(join(path, 'meta_data.json'), 'r') as json_file:
        meta_data = json.load(json_file)
    # removing old factorization with same name (if exists)
    delete_factorization_by_name(name, path)
    prefix = 'mat_' if is_matrix else 'ten_'
    ann_path = join(path, prefix + str(dimensions) + '_1_' + name)
    create_folder_if_absent(ann_path)
    # factorizing the data
    start = time.time()
    result = train_embeddings(path, dimensions, name, neg_ratio, min_freq,
                              is_matrix, max_epochs, batch_size, learning_rate)
    t = AnnoyIndex(dimensions,  'angular')
    if is_matrix:
        ext_embds = result
    else:
        mod_embds, asp_embds = result
        # loading meta data and building indices
        modifiers, aspects = meta_data['mod_index'], meta_data['asp_index']
        mod2idx = dict(zip(modifiers, range(len(modifiers))))
        asp2idx = dict(zip(aspects, range(len(aspects))))
        extractions = pd.read_csv(join(path, 'extr_index.csv'))
        ext_embds = []
        for j, row in extractions.iterrows():
            m, a = row['modifier'], row['aspect']
            ext_embds.append(np.multiply(mod_embds[mod2idx[m]],
                                         asp_embds[asp2idx[a]]))
    for j, embd in enumerate(ext_embds):
        t.add_item(j, embd)
    t.build(100)
    t.save(join(ann_path, 'embd_0.ann'))
    end = time.time()
    print('Factorization completed in %d seconds' % (end - start))


def train_embeddings(path, dimensions, name, neg_ratio, min_freq, is_matrix,
                     max_epochs, batch_size, learning_rate):
    data = NDArrayDataset(path, neg_ratio, min_freq, is_matrix)
    data_loader = DataLoader(data, batch_size=batch_size)
    if is_matrix:
        num_rows, num_exts = data_loader.dataset.ndarray.shape
        model = UniversalSchemaModel(num_rows, num_exts, dimensions)
    else:
        num_rows, num_mods, num_asps = data_loader.dataset.ndarray.shape
        model = UniversalSchema3DModel(num_rows, num_mods, num_asps, dimensions)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    # The training loop
    for epoch in range(max_epochs):
        losses = []
        for batch, labels in data_loader:
            model.zero_grad()
            pred = model(torch.autograd.Variable(torch.LongTensor(batch)))
            loss = criterion(pred, labels.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.numpy())
        print("Epoch: {}, avg_batch_loss: {}".
              format(epoch, sum(losses) / len(losses)))
    # Returning the results
    if is_matrix:
        return model.E.weight.data.numpy()
    return model.M.weight.data.numpy(), model.A.weight.data.numpy()


def main():
    args = parse_arguments()
    run_universal_schema(**vars(args))


if __name__ == "__main__":
    main()
