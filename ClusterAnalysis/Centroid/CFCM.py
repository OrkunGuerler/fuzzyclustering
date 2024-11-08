import numpy as np
import numpy.typing as npt
from ClusterAnalysis._auxiliary.distance import pairwise_sqrd_euclidean_distance, euclidean_distance
from ClusterAnalysis import FCM


def pairwise_k_nearest_neighbors(x: npt.NDArray, k: int) -> npt.NDArray:
    distances = pairwise_sqrd_euclidean_distance(x, x)
    k_indices = np.argsort(distances)[:, 1:k+1]
    return x[k_indices]


def calculate_credibilities(dataset: npt.NDArray, n_clusters: int,
                            epsilon: float) -> npt.NDArray:
    n_points, _ = dataset.shape
    k = int(np.ceil(epsilon * (n_points / n_clusters)))
    knn = pairwise_k_nearest_neighbors(dataset, k)
    mu = np.zeros(n_points)
    for j in range(n_points):
        dists = pairwise_sqrd_euclidean_distance(knn[j], dataset[j])
        mu[j] = np.sum(dists) / k
    credibilities = np.zeros(n_points)
    for j in range(n_points):
        numer = mu[j] - np.min(mu)
        denom = np.max(mu) - np.min(mu)
        credibilities[j] = 1 - (numer / denom)
    return credibilities


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


def credibilistic_fuzzy_c_means(dataset: npt.NDArray, n_clusters: int,
                                epsilon: float = 0.5, fuzziness: float = 2,
                                tolerance: float = 1e-4, max_iter: int = 10000
                                ) -> tuple[npt.NDArray, npt.NDArray]:
    n_points, n_dims = dataset.shape

    # initialisation of weights
    weights = np.random.dirichlet(np.ones(n_clusters), size=n_points)

    # calculation of credibilities of the dataset
    credibilities = calculate_credibilities(dataset, n_clusters, epsilon)

    # initialisation of centroids
    prev_centroids = np.zeros((n_clusters, n_dims))

    for _ in range(max_iter):
        # calculation of new centroids according to the weights of the dataset
        centroids = FCM.update_centroids(dataset, weights, fuzziness)

        # updating the weights of the dataset according to their distances to the new centroids
        weights = np.multiply(FCM.calculate_weights(dataset, centroids, fuzziness),
                              credibilities[:, np.newaxis])

        # check for convergence
        if euclidean_distance(prev_centroids, centroids) < tolerance:
            break
        prev_centroids = centroids
    return {"weights": weights, "centroids": centroids}
