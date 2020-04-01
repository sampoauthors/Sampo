import time
import json
import spacy
import sparse
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from annoy import AnnoyIndex
from hottbox.core import Tensor
from hottbox.algorithms.decomposition.fusion import CMTF

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
    parser.add_argument('--name', default='current' ,
                        help='The name associated with the factorization \
                        results. This name can be used later for evalution \
                        and other analysis on the results.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gauss', action='store_true',
                       help='Add gaussian noise to the tensor')
    group.add_argument('--poisson', action='store_true',
                       help='Add poisson noise to the tensor')
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


def build_coupled_matrix(nlp, spans):
    lang_embd = []
    for s in tqdm(spans):
        lang_embd.append(nlp(s).vector)
    return np.array(lang_embd)


def factorize_matrices(matrices, iterations, dimensions, mat_path, path):
    # loading additional data required to build coupled matrices
    ext_ind = pd.read_csv(join(path, 'extr_index.csv'))
    extractions = list(ext_ind.apply(lambda x: (x['modifier'] + ' ' +
                                                x['aspect']), axis=1))
    # creating the coupled matrices
    nlp = spacy.load('en_core_web_md')
    dim1, dim2 = matrices[0].shape
    dim1_mat = Tensor(np.zeros((dim1, 0)))     # fake matrix with no columns
    dim2_mat = Tensor(build_coupled_matrix(nlp, extractions))
    # factorizing matrices
    for i, mat in tqdm(enumerate(matrices), total=iterations):
        hb_mat = Tensor(mat)
        cmtf = CMTF(random_state=0)
        (factors, _, _, _) = cmtf.decompose(hb_mat, [dim1_mat, dim2_mat],
                                            (dimensions,))
        # getting extraction embeddings
        ext_embds = factors[1]
        # computing and indexing the embedding of each extraction
        t = AnnoyIndex(dimensions, 'angular')
        for j, embd in enumerate(ext_embds):
            t.add_item(j, embd)
        t.build(100)
        t.save(join(mat_path, 'embd_' + str(i) + '.ann'))


def factorize_tensors(tensors, iterations, dimensions, ten_path,
                      meta_data, path):
    # loading additional data required to build coupled matrices
    modifiers = meta_data['mod_index']
    mod2idx = dict(zip(modifiers, range(len(modifiers))))
    aspects = meta_data['asp_index']
    asp2idx = dict(zip(aspects, range(len(aspects))))
    extractions = pd.read_csv(join(path, 'extr_index.csv'))
    # creating the coupled matrices
    nlp = spacy.load('en_core_web_md')
    dim1, dim2, dim3 = tensors[0].shape
    dim1_mat = Tensor(np.zeros((dim1, 0)))     # fake matrix with no columns
    dim2_mat = Tensor(build_coupled_matrix(nlp, modifiers))
    dim3_mat = Tensor(build_coupled_matrix(nlp, aspects))
    # factorizing tensors
    for i, ten in tqdm(enumerate(tensors), total=iterations):
        cmtf = CMTF(random_state=0)
        (factors, _, _, _) = cmtf.decompose(
            ten, [dim1_mat, dim2_mat, dim3_mat], (dimensions,))
        # getting modifier and aspect embeddings
        mod_embds, asp_embds = factors[1], factors[2]
        # computing and indexing the embedding of each extraction
        t = AnnoyIndex(dimensions, 'angular')
        for j, row in extractions.iterrows():
            m, a= row['modifier'], row['aspect']
            embd = np.multiply(mod_embds[mod2idx[m]], asp_embds[asp2idx[a]])
            t.add_item(j, embd)
        t.build(100)
        t.save(join(ten_path, 'embd_' + str(i) + '.ann'))


def run_cmtf(path, dimensions=20, is_matrix=True, name='current',
             noise='orig', iterations=1, fixed='False'):
    # checking if the arguments are valid
    assert is_ndarray_folder(path), "Not a valid path."
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
        factorize_matrices(matrices, iterations, dimensions, mat_path, path)
    else:
        tensor = sparse.load_npz(join(path, 'tensor.npz')).todense()
        tensors = prepare_ndarrays(tensor, iterations, fixed, noise)
        create_folder_if_absent(ten_path)
        factorize_tensors(tensors, iterations, dimensions, ten_path,
                          meta_data, path)
    end = time.time()
    print('Factorization completed in %d seconds' % (end - start))


def main():
    args = parse_arguments()
    args.is_matrix = args.matrix
    args.noise = 'gauss' if args.gauss else 'orig'
    args.noise = 'poisson' if args.poisson else args.noise
    del args.matrix
    del args.tensor
    del args.gauss
    del args.poisson
    run_cmtf(**vars(args))


if __name__ == "__main__":
    main()
