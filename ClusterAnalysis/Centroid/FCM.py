import numpy as np
import numpy.typing as npt
from ClusterAnalysis._auxiliary.distance import euclidean_distance, pairwise_sqrd_euclidean_distance


def update_centroids(dataset: npt.NDArray, weights: npt.NDArray,
                     fuzziness: float) -> npt.NDArray:
    n_points, n_dims = dataset.shape
    _, n_clusters = weights.shape
    centroids = np.zeros((n_clusters, n_dims))
    f_weights = np.power(weights, fuzziness)
    for i in range(n_clusters):
        numer, denom = 0, 0
        for j in range(n_points):
            numer += f_weights[j, i] * dataset[j]
            denom += f_weights[j, i]
        centroids[i] = numer / denom
    return centroids


def calculate_weights(dataset: npt.NDArray, centroids: npt.NDArray,
                      fuzziness: float) -> npt.NDArray:
    n_points, _ = dataset.shape
    n_clusters, _ = centroids.shape
    weights = np.zeros((n_points, n_clusters))
    dists = pairwise_sqrd_euclidean_distance(dataset, centroids)
    for i in range(n_clusters):
        for j in range(n_points):
            summ = 0
            for k in range(n_clusters):
                summ += (dists[j, i] / dists[j, k]) ** (1 / (fuzziness - 1))
            weights[j, i] = 1 / summ
    return weights


def fuzzy_c_means(dataset: npt.NDArray, n_clusters: int, fuzziness: float = 2,
                  tolerance: float = 1e-4, max_iter: int = 10000
                  ) -> dict[npt.NDArray, npt.NDArray]:
    # get size of the dataset
    n_points, n_dims = dataset.shape

    # initialisation of weights
    weights = np.random.dirichlet(np.ones(n_clusters), size=n_points)

    # initialisation of centroids
    prev_centroids = np.zeros((n_clusters, n_dims))

    for _ in range(max_iter):
        # calculation of new centroids according to the weights of the dataset
        centroids = update_centroids(dataset, weights, fuzziness)

        # updating the weights of the dataset according to their distances to the new centroids
        weights = calculate_weights(dataset, centroids, fuzziness)

        # check for convergence
        if euclidean_distance(prev_centroids, centroids) < tolerance:
            break
        prev_centroids = centroids
    return {"weights": weights, "centroids": centroids}
