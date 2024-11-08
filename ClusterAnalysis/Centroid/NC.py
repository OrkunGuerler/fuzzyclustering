import numpy as np
import numpy.typing as npt
from ClusterAnalysis._auxiliary.distance import euclidean_distance, pairwise_sqrd_euclidean_distance
from ClusterAnalysis import FCM


def noise_cluster_distance(dataset: npt.NDArray, centroids: npt.NDArray,
                           multiplier: float) -> npt.NDArray:
    n_points, _ = dataset.shape
    n_clusters, _ = centroids.shape
    sum_of_dists = np.sum(pairwise_sqrd_euclidean_distance(dataset, centroids))
    return (multiplier / (n_points * (n_clusters))) * sum_of_dists


def calculate_weights(dataset: npt.NDArray, centroids: npt.NDArray,
                      multiplier, fuzziness: float) -> npt.NDArray:
    n_points, _ = dataset.shape
    n_clusters, _ = centroids.shape
    weights = np.zeros((n_points, n_clusters))
    dists = pairwise_sqrd_euclidean_distance(dataset, centroids)
    noise_distance = noise_cluster_distance(dataset, centroids, multiplier)
    outliers = (dists / noise_distance) ** (1 / (fuzziness - 1))
    for i in range(n_clusters):
        for j in range(n_points):
            summ = 0
            for k in range(n_clusters):
                summ += (dists[j, i] / dists[j, k]) ** (1 / (fuzziness - 1))
            weights[j, i] = 1 / (summ + outliers[j, i])
    return weights


def noise_clustering(dataset: npt.NDArray, n_clusters: int,
                     multiplier: float = 0.2525, fuzziness: float = 2,
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
        centroids = FCM.update_centroids(dataset, weights, fuzziness)

        # updating the weights of the dataset according to their distances to the new centroids
        weights = calculate_weights(dataset, centroids, multiplier, fuzziness)

        # check for convergence
        if euclidean_distance(prev_centroids, centroids) < tolerance:
            break
        prev_centroids = centroids
    return {"weights": weights, "centroids": centroids}
