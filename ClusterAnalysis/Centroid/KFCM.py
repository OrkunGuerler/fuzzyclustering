import numpy as np
import numpy.typing as npt
from ClusterAnalysis._auxiliary.distance import pairwise_euclidean_distance, rbf_kernel_distance, euclidean_distance
from ClusterAnalysis import FCM


def modified_distance_variance(dataset: npt.NDArray) -> npt.NDArray:
    n_points, _ = dataset.shape
    v = np.sum(dataset, axis=0) / n_points
    v_dist = pairwise_euclidean_distance(dataset, v)
    d = np.sum(v_dist, axis=0) / n_points
    return np.sum((v_dist - d) ** 2) / (n_points - 1)


def update_centroids(dataset: npt.NDArray, bandwidth: float,
                     weights: npt.NDArray, prev_centroids: npt.NDArray,
                     fuzziness: float) -> npt.NDArray:
    n_points, n_dims = dataset.shape
    _, n_clusters = weights.shape
    centroids = np.zeros((n_clusters, n_dims))
    f_weights = np.power(weights, fuzziness)
    for i in range(n_clusters):
        numer, denom = 0, 0
        for j in range(n_points):
            dist = rbf_kernel_distance(
                dataset[j], prev_centroids[i], bandwidth)
            numer += f_weights[j, i] * dist * dataset[j]
            denom += f_weights[j, i] * dist
        centroids[i] = numer / denom
    return centroids


def calculate_weights(dataset: npt.NDArray, bandwidth: float,
                      centroids: npt.NDArray, fuzziness: float
                      ) -> npt.NDArray:
    n_points, _ = dataset.shape
    n_clusters, _ = centroids.shape
    weights = np.zeros((n_points, n_clusters))
    for i in range(n_clusters):
        for j in range(n_points):
            summ = 0
            d_ji = rbf_kernel_distance(dataset[j], centroids[i], bandwidth)
            for k in range(n_clusters):
                d_jk = rbf_kernel_distance(dataset[j], centroids[k], bandwidth)
                summ += ((1 - d_ji) / (1 - d_jk)) ** (1 / (fuzziness - 1))
            weights[j, i] = 1 / summ
    return weights


def kernel_fuzzy_c_means(dataset: npt.NDArray, n_clusters: int, fuzziness: float = 2,
                         tolerance: float = 1e-4, max_iter: int = 10000
                         ) -> tuple[npt.NDArray, npt.NDArray]:
    # get size of the dataset
    n_points, _ = dataset.shape

    # calculation of variance of the dataset
    bandwidth = np.sqrt(modified_distance_variance(dataset))

    # initialisation of weights
    weights = np.random.dirichlet(np.ones(n_clusters), size=n_points)

    # initialisation of centroids
    prev_centroids = FCM.update_centroids(dataset, weights, fuzziness)

    for _ in range(max_iter):
        # calculation of new centroids according to the weights of the dataset
        centroids = update_centroids(dataset, bandwidth, weights,
                                     prev_centroids, fuzziness)

        # updating the weights of the dataset according to their distances to the new centroids
        weights = calculate_weights(dataset, bandwidth, centroids, fuzziness)

        # check for convergence
        if euclidean_distance(prev_centroids, centroids) < tolerance:
            break
        prev_centroids = centroids
    return {"weights": weights, "centroids": centroids}
