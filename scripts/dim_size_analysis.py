import sparse
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from tabulate import tabulate
from tensorly.decomposition import parafac
from tensorly.decomposition import non_negative_parafac as nn_parafac
from tensorly import kruskal_to_tensor

from sampo.os_utils import is_ndarray_folder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path to the folder that contains the \
                        tensor/matrix to be factorized')
    parser.add_argument('-m', '--max_dim', type=int, default=50,
                        help='The maximum embedding dimension to be tried')
    parser.add_argument('-n', '--nonnegative', action='store_true',
                        help='Use non-negative tensor factorization')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tensor', '--3d', action='store_true',
                       help='Factorize only the tensor.')
    group.add_argument('--matrix', '--2d', action='store_true',
                       help='Factorize only the matrix.')
    parser.add_argument('-i', '--iterations', type=int, default=1,
                        help='Number of repetitions of factorization')
    args = parser.parse_args()
    return args


def get_factorization_error(ndarray, args):
    factorizer = nn_parafac if args.nonnegative else parafac
    (weights, factors), errors = \
        factorizer(ndarray, rank=args.dimensions, return_errors=True)
    rec = kruskal_to_tensor((weights, factors))
    max_err = np.max(np.absolute(rec - ndarray))
    mean_abs_err = np.mean(np.absolute(rec - ndarray))
    mean_sq_err = np.mean(np.multiply(rec - ndarray, rec - ndarray))
    return errors[-1], max_err, mean_abs_err, mean_sq_err


def main():
    args = parse_arguments()
    # checking if the arguments are valid
    assert is_ndarray_folder(args.path), "Invalid path."
    # factorizing the data
    dims = [1] + list(range(5, args.max_dim, 5)) + [args.max_dim]
    table = []
    if args.matrix:
        ndarray = sparse.load_npz(join(args.path, 'matrix.npz')).todense()
    else:
        ndarray = sparse.load_npz(join(args.path, 'tensor.npz')).todense()
    for d in tqdm(dims):
        err_sum, max_err_sum, mean_abs_err_sum, mean_sq_err_sum = 0, 0, 0, 0
        args.dimensions = d
        for _ in range(args.iterations):
            err, max_err, mean_abs_err, mean_sq_err = \
                get_factorization_error(ndarray, args)
            err_sum += err
            max_err_sum += max_err
            mean_abs_err_sum += mean_abs_err
            mean_sq_err_sum += mean_sq_err
        table.append([d, err_sum / args.iterations,
                      max_err_sum / args.iterations,
                      mean_abs_err_sum / args.iterations,
                      mean_sq_err_sum / args.iterations])
    print(tabulate(table, headers=['Dim', 'Err.', 'Max Err.',
                                   'Mean Abs. Err.', 'Mean Sq. Err.'],
                   tablefmt='orgtbl'))


if __name__ == "__main__":
    main()
