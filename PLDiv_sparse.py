import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree

def PLDiv_Sparse_MST(distance_matrix, sparse = 0.5):
    lambdas = getGreedyPerm(distance_matrix)
    DSparse = getApproxSparseDM(lambdas, eps=sparse, D=distance_matrix)

    # Use MST-based fast approximation
    PLDiv_val = fast_PLDiv_approx(DSparse)
    return PLDiv_val



def fast_PLDiv_approx(D, overwrite= True): 
    """
    Computes PLDiv from a distance matrix (sparse or dense)
    without allocating new dense memory.
    """
    mst = minimum_spanning_tree(D, overwrite=overwrite)
    val = 0.25 * np.sum(mst.data ** 2)
    
    return val


from ripser import ripser
from persim import plot_diagrams

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse
import time

def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points

    Return
    ------
    lamdas: list
        Insertion radii of all points
    """

    N = D.shape[0]

    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]


def getApproxSparseDM(lambdas, eps, D):
    """
    Purpose: To return the sparse edge list with the warped distances, sorted by weight

    Parameters
    ----------
    lambdas: list
        insertion radii for points
    eps: float
        epsilon approximation constant
    D: ndarray
        NxN distance matrix, okay to modify because last time it's used

    Return
    ------
    DSparse: scipy.sparse
        A sparse NxN matrix with the reweighted edges
    """
    N = D.shape[0]
    E0 = (1+eps)/eps
    E1 = (1+eps)**2/eps

    nBounds = ((eps**2+3*eps+2)/eps)*lambdas

    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf)*(idx == 1)]
    J = J[(D < np.inf)*(idx == 1)]
    D = D[(D < np.inf)*(idx == 1)]

    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    M = np.minimum((E0 + E1)*minlam, E0*(minlam + maxlam))

    t = np.arange(len(I))
    t = t[D <= M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]

    t = np.ones(len(I))

    t[D <= 2*minlam*E0] = 0

    D[t == 1] = 2.0*(D[t == 1] - minlam[t == 1]*E0) 
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()