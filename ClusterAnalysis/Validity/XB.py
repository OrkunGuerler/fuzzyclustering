import numpy as np
import numpy.typing as npt
from ClusterAnalysis._auxiliary.distance import pairwise_sqrd_euclidean_distance

def xie_beni(dataset: npt.NDArray, weights: npt.NDArray, centroids: npt.NDArray,
             fuzziness: float = 2) -> npt.NDArray:
    """Xie/Beni Index: Measures the compactness and separation of the clusters. It considers the distances between data points and their cluster centers as well as the distances between different cluster centers.
    Lower values are better; means compact and well-separated clusters.

    Args:
        dataset (npt.NDArray): _description_
        weights (npt.NDArray): _description_
        centroids (npt.NDArray): _description_
        fuzziness (float, optional): _description_. Defaults to 2.

    Returns:
        npt.NDArray: _description_
    """
    x_dists = pairwise_sqrd_euclidean_distance(dataset, centroids)
    v_dists = pairwise_sqrd_euclidean_distance(centroids, centroids)
    v_dists[np.diag_indices_from(v_dists)] = np.inf
    return np.sum(x_dists * np.power(weights, fuzziness)) / (dataset.shape[0] * np.min(v_dists))
