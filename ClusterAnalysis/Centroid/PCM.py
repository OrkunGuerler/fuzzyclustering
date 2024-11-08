import numpy as np
import numpy.typing as npt
from ClusterAnalysis import FCM, fuzzy_c_means
from ClusterAnalysis._auxiliary.distance import pairwise_sqrd_euclidean_distance, euclidean_distance


def calculate_bandwidth(dataset: npt.NDArray,
                        weights: npt.NDArray, centroids: npt.NDArray,
                        fuzziness: int, k: float) -> npt.NDArray:
    n_points, n_clusters = weights.shape
    bandwidth = np.zeros(n_clusters)
    f_weights = np.power(weights, fuzziness)
    dists = pairwise_sqrd_euclidean_distance(dataset, centroids)
    for i in range(n_clusters):
        numer, denom = 0, 0
        for j in range(n_points):
            numer += f_weights[j, i] * dists[j, i]
            denom += f_weights[j, i]
        bandwidth[i] = k * (numer / denom)
    return bandwidth


def calculate_typicalities(dataset: npt.NDArray, centroids: npt.NDArray,
                           bandwidth: npt.NDArray, belongingness: float, b: float = 1
                           ) -> npt.NDArray:
    n_points, _ = dataset.shape
    n_clusters, _ = centroids.shape
    typicals = np.zeros((n_points, n_clusters))
    dists = pairwise_sqrd_euclidean_distance(dataset, centroids)
    for i in range(n_clusters):
        for j in range(n_points):
            mid = (b * dists[j, i]) / bandwidth[i]
            power = 1 / (belongingness - 1)
            typicals[j, i] = 1 / (1 + (mid ** power))
    return typicals


def possibilistic_c_means(dataset: npt.NDArray, n_clusters: int,
                          fuzziness: float = 2, belongingness: float = 2,  k: float = 1,
                          tolerance: float = 1e-4, max_iter: int = 10000
                          ) -> dict[npt.NDArray, npt.NDArray]:
    # initialisation of weights and centroids
    fcm = fuzzy_c_means(dataset, n_clusters, fuzziness,
                        tolerance, max_iter)
    weights, prev_centroids = fcm.get("weights"), fcm.get("centroids")

    # calculation of the bandwidth
    bandwidth = calculate_bandwidth(
        dataset, weights, prev_centroids, fuzziness, k)

    for _ in range(max_iter):
        # updating the weights of the dataset according to their distances to the new centroids
        typicalities = calculate_typicalities(
            dataset, prev_centroids, bandwidth, belongingness)

        # calculation of new centroids according to the weights of the dataset
        centroids = FCM.update_centroids(
            dataset, typicalities, belongingness)

        # check for convergence
        if euclidean_distance(prev_centroids, centroids) < tolerance:
            break
        prev_centroids = centroids
    return {"weights": weights, "typicalities": typicalities, "centroids": centroids}
