import numpy as np
import numpy.typing as npt
from ClusterAnalysis._auxiliary.distance import pairwise_sqrd_euclidean_distance

def fuzzy_silhouette(dataset: npt.NDArray, weights: npt.NDArray) -> npt.NDArray:
    """Fuzzy Silhouette Index: Combines membership degrees with the silhouette value, which measures how similar an object is to its own cluster compared to other clusters.
    Higher values are better; means well-defined clusters.

    Args:
        dataset (npt.NDArray): _description_
        w (npt.NDArray): _description_

    Returns:
        npt.NDArray: returns fs value
    """
    n_points, n_clusters = weights.shape
    dists = pairwise_sqrd_euclidean_distance(dataset, dataset)

    a = np.empty(n_points)
    for j in range(n_points):
        aj = []
        for i in range(n_clusters):
            numer, denom = 0, 0
            for k in range(n_points):
                if j != k:
                    numer += (weights[j, i] and weights[k, i]) * dists[j, k]
                    denom += (weights[j, i] and weights[k, i])
            aj.append(numer / denom)
        a[j] = np.min(np.array(aj))

    b = np.empty(n_points)
    for j in range(n_points):
        bj = []
        for r in range(n_clusters - 1):
            numer, denom = 0, 0
            for s in range(r + 1, n_clusters):
                for k in range(n_points):
                    if j != k:
                        lhs = (weights[j, r] and weights[k, s])
                        rhs = (weights[j, s] and weights[k, r])
                        numer += (lhs or rhs) * dists[j, k]
                        denom += (lhs or rhs)
            bj.append(numer/denom)
        b[j] = np.min(np.array(bj))

    s = np.empty(n_points)
    for j in range(n_points):
        s[j] = (b[j] - a[j]) / np.max((a[j], b[j]))

    return np.sum(s) / n_points