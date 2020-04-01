import time
import json
import sparse
import argparse
import numpy as np
import pandas as pd
import tensorly as tl
from tqdm import tqdm
from os.path import join
from annoy import AnnoyIndex
from tensorly.decomposition import parafac
from tensorly.decomposition import non_negative_parafac as nn_parafac

from sampo.os_utils import is_ndarray_folder
from sampo.noise_utils import prepare_ndarrays
from sampo.os_utils import create_folder_if_absent, delete_factorization_by_name


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path to the folder that contains the \
                        tensor/matrix to be factorized')
    parser.add_argument('-d', '--dimensions', type=int, default=20,
                        help='The desired embedding dimension (i.e., rank)')
    parser.add_argument('-n', '--nonnegative', action='store_true',
                        help='Use non-negative tensor factorization')
    parser.add_argument('--name', default='current' ,
                        help='The name associated with the factorization \
                        results. This name can be used later for evalution \
                        and other analysis on the results.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use pytorch as the backend and gpus (if any).')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gauss', action='store_true',
                       help='Add gaussian noise to the tensor')
    group.add_argument('--poisson', action='store_true',
                       help='Add poisson noise to the tensor')
    group.add_argument('--gen', action='store_true',
                       help='Add gaussian noise using the generative model')
    parser.add_argument('-i', '--iterations', type=int, default=1,
                        help='Number of repetitions of factorization')
    parser.add_argument('--fixed', action='store_true',
                        help='Fix the noisy tensor for repetitions.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tensor', '--3d', action='store_true',
                       help='Factorize only the tensor.')
    group.add_argument('--matrix', '--2d', action='store_true',
                       help='Factorize only the matrix.')
    args = parser.parse_args()
    return args


def factorize_matrices(matrices, iterations, dimensions, nonnegative,
                       mat_path, cuda):
    for i, mat in tqdm(enumerate(matrices), total=iterations):
        mat = tl.tensor(mat, device="cuda:0") if cuda else mat
        factorizer = nn_parafac if nonnegative else parafac
        weights, factors = factorizer(mat, rank=dimensions, init='random')
        # getting modifier and aspect embeddings
        ext_embds = factors[1].cpu().numpy() if cuda else factors[1]
        # computing and indexing the embedding of each extraction
        t = AnnoyIndex(dimensions, 'angular')
        for j, embd in enumerate(ext_embds):
            t.add_item(j, embd)
        t.build(100)
        t.save(join(mat_path, 'embd_' + str(i) + '.ann'))


def factorize_tensors(tensors, iterations, dimensions, nonnegative, ten_path,
                      meta_data, path, cuda):
    # loading meta data and building indices
    modifiers = meta_data['mod_index']
    mod2idx = dict(zip(modifiers, range(len(modifiers))))
    aspects = meta_data['asp_index']
    asp2idx = dict(zip(aspects, range(len(aspects))))
    extractions = pd.read_csv(join(path, 'extr_index.csv'))
    # factorizing the tensors
    for i, ten in tqdm(enumerate(tensors), total=iterations):
        ten = tl.tensor(ten, device="cuda:0") if cuda else ten
        factorizer = nn_parafac if nonnegative else parafac
        weights, factors = factorizer(ten, rank=dimensions)
        # getting modifier and aspect embeddings
        mod_embds, asp_embds = factors[1], factors[2]
        # computing and indexing the embedding of each extraction
        t = AnnoyIndex(dimensions, 'angular')
        for j, row in extractions.iterrows():
            m, a = row['modifier'], row['aspect']
            if cuda:
                mvec = mod_embds[mod2idx[m]].cpu().numpy()
                avec = asp_embds[asp2idx[a]].cpu().numpy()
            else:
                mvec = mod_embds[mod2idx[m]]
                avec = asp_embds[asp2idx[a]]
            embd = np.multiply(mvec, avec)
            t.add_item(j, embd)
        t.build(100)
        t.save(join(ten_path, 'embd_' + str(i) + '.ann'))


def run_parafac(path, dimensions=20, nonnegative=False, is_matrix=True,
                name='current', cuda=False, noise='orig', iterations=1,
                fixed='False'):
    # setting up  the backend
    if cuda:
        tl.set_backend('pytorch')
    # checking if the arguments are valid
    assert is_ndarray_folder(path), "Invalid path."
    assert '_' not in name, "Name cannot contain the '_' symbol."
    # creating some useful paths to store factorization results
    mat_path = join(path, 'mat_' + str(dimensions) +
                    '_' + str(iterations) + '_' + name)
    ten_path = join(path, 'ten_' + str(dimensions) +
                    '_' + str(iterations) + '_' + name)
    # loading the meta data
    with open(join(path, 'meta_data.json'), 'r') as json_file:
        meta_data = json.load(json_file)
    # removing old factorization with same name (if exists)
    delete_factorization_by_name(name, path)
    # factorizing the data
    start = time.time()
    if is_matrix:
        matrix = sparse.load_npz(join(path, 'matrix.npz')).todense()
        matrices = prepare_ndarrays(matrix, iterations, fixed, noise)
        create_folder_if_absent(mat_path)
        factorize_matrices(matrices, iterations, dimensions,
                           nonnegative, mat_path, cuda)
    else:
        tensor = sparse.load_npz(join(path, 'tensor.npz')).todense()
        tensors = prepare_ndarrays(tensor, iterations, fixed, noise)
        create_folder_if_absent(ten_path)
        factorize_tensors(tensors, iterations, dimensions, nonnegative,
                          ten_path, meta_data, path, cuda)
    end = time.time()
    print('Factorization completed in %d seconds' % (end - start))


def main():
    args = parse_arguments()
    args.is_matrix = args.matrix
    args.noise = 'gauss' if args.gauss else 'orig'
    args.noise = 'poisson' if args.poisson else args.noise
    if args.gen:
        mode = 'gpu' if args.cuda else 'cpu'
        args.noise = 'gen' + mode + str(args.dimensions)
    del args.matrix
    del args.tensor
    del args.gauss
    del args.gen
    del args.poisson
    run_parafac(**vars(args))


if __name__ == "__main__":
    main()
