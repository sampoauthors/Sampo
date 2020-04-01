import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac


def prepare_ndarrays(ndarray, iterations, fixed, noise):
    tensors, base = [], None
    if noise.startswith('gen'):
        cuda = noise[3:6] == 'gpu'
        dim = int(noise[6:])
        # Factorizing the tensor
        ndarray = tl.tensor(ndarray, device="cuda:0") if cuda else ndarray
        weights, factors = parafac(ndarray, rank=dim, init='random')
        base = tl.kruskal_to_tensor((weights, factors))
        noise_tensor = (ndarray - base).flatten()
        print('Estimated noise mean: {}'.format(np.mean(noise_tensor)))
        print('Estimated noise std: {}'.format(np.std(noise_tensor)))
    curr_tensor = get_noisy_tensor(ndarray, noise, base)
    for _ in range(iterations):
        if not fixed:
            curr_tensor = get_noisy_tensor(ndarray, noise, base)
        tensors.append(curr_tensor)
    return tensors


def gauss(tensor, mu=0, sigma=0.01):
    # Gaussian-distributed additive noise.
    shape = tensor.shape
    return tensor + np.random.normal(mu, sigma, shape)


def poisson(tensor):
    # Poisson-distributed noise generated from the data.
    vals = len(np.unique(tensor))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(tensor * vals) / float(vals)
    return noisy


def generative(tensor, base):
    # Calculating the noise
    noise_tensor = (tensor - base).flatten()
    noise_mean = np.mean(noise_tensor)
    noise_std = np.std(noise_tensor)
    return gauss(base, noise_mean, noise_std)


def get_noisy_tensor(tensor, noise, base=None):
    if noise == 'orig':
        return tensor
    elif noise == 'gauss':
        return gauss(tensor, mu=0, sigma=1)
    elif noise == 'poisson':
        return poisson(tensor)
    elif noise.startswith('gen'):
        return generative(tensor, base)
    #TODO:: add more noise models for laplacian and gaussian mixture
    return None
