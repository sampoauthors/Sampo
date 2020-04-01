import os
import shutil


def create_folder_if_absent(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def get_subfolders(path):
    subfolders = [os.path.join(path, x) for x in os.listdir(path)
                  if os.path.isdir(os.path.join(path, x))]
    return subfolders


def is_factorization_folder(path):
    files = [x for x in os.listdir(path)]
    return 'embd_0.ann' in files


def is_ndarray_folder(path):
    files = [x for x in os.listdir(path)]
    return ('matrix.npz' in files) and ('tensor.npz' in files)


def get_factorization_path_by_name(name, ndarray_path):
    subfolders = get_subfolders(ndarray_path)
    names = [x.split('_')[-1] for x in subfolders]
    try:
        idx = names.index(name)
        return subfolders[idx]
    except ValueError:
        return None


def get_all_factorization_names(ndarray_path):
    subfolders = get_subfolders(ndarray_path)
    names = [x.split('_')[-1] for x in subfolders if is_factorization_folder(x)]
    return names


def get_factorization_params_from_path(factorization_path):
    folder = os.path.basename(os.path.normpath(factorization_path))
    dim, embd_dim, iterations, name = folder.split('_')
    return dim, int(embd_dim), int(iterations), name


def delete_factorization_by_name(name, ndarray_path):
    factorization_path = get_factorization_path_by_name(name, ndarray_path)
    if factorization_path:
        shutil.rmtree(factorization_path)
