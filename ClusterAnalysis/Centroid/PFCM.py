import numpy as np
import numpy.typing as npt
from ClusterAnalysis._auxiliary.distance import euclidean_distance
from ClusterAnalysis.Centroid import PCM, FCM, fuzzy_c_means


def calculate_centroids(dataset: npt.NDArray,
                        weights: npt.NDArray, typicalities: npt.NDArray,
                        fuzziness: float, belongingness: float,
                        a: float, b: float) -> npt.NDArray:
    n_points, n_dims = dataset.shape
    _, n_clusters = weights.shape
    centroids = np.zeros((n_clusters, n_dims))
    f_rate = a * np.power(weights, fuzziness)
    b_rate = b * np.power(typicalities, belongingness)
    r_weights = f_rate + b_rate
    for i in range(n_clusters):
        numer, denom = 0, 0
        for j in range(n_points):
            numer += r_weights[j, i] * dataset[j]
            denom += r_weights[j, i]
        centroids[i] = numer / denom
    return centroids


def possibilistic_fuzzy_c_means(dataset: npt.NDArray, n_clusters: int,
                                fuzziness: float = 2, belongingness: float = 2,
                                k: float = 1, a: float = 0.5, b: float = 0.5,
                                tolerance: float = 1e-4, max_iter: int = 10000
                                ) -> dict[npt.NDArray, npt.NDArray, npt.NDArray]:
    # initialisation of weights and centroids
    fcm = fuzzy_c_means(dataset, n_clusters, fuzziness,
                        tolerance, max_iter)
    weights, prev_centroids = fcm.get("weights"), fcm.get("centroids")

    # calculation of the bandwidth
    bandwidth = PCM.calculate_bandwidth(
        dataset, weights, prev_centroids, fuzziness, k)

    for _ in range(max_iter):
        # updating the weights of the dataset according to their distances to the new centroids
        typicalities = PCM.calculate_typicalities(
            dataset, prev_centroids, bandwidth, belongingness, b)

        # updating the weights of the dataset according to their distances to the new centroids
        weights = FCM.calculate_weights(dataset, prev_centroids, fuzziness)

        # calculation of new centroids according to the weights of the dataset
        centroids = calculate_centroids(dataset, weights, typicalities,
                                        fuzziness, belongingness, a, b)

        # check for convergence
        if euclidean_distance(prev_centroids, centroids) < tolerance:
            break
        prev_centroids = centroids
    return {"weights": weights, "typicalities": typicalities, "centroids": centroids}
