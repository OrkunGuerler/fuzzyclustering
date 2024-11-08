import numpy as np
import numpy.typing as npt
from ClusterAnalysis import FCM
from ClusterAnalysis._auxiliary.distance import mahalanobis_distance, euclidean_distance


def fuzzy_covariance_matrix(dataset: npt.NDArray, weights: npt.NDArray,
                            centroids: npt.NDArray, volumes: npt.NDArray,
                            fuzziness: float
                            ) -> npt.NDArray:
    n_points, n_dims = dataset.shape
    _, n_clusters = weights.shape
    f_weights = np.power(weights, fuzziness)
    f = np.zeros((n_clusters, n_dims, n_dims))
    a = np.zeros((n_clusters, n_dims, n_dims))
    for i in range(n_clusters):
        numer, denom = 0, 0
        for j in range(n_points):
            diff = np.atleast_2d(dataset[j] - centroids[i])
            numer += f_weights[j, i] * np.dot(diff.T, diff)
            denom += f_weights[j, i]
        f[i] = numer / denom
        a[i] = np.dot(np.power(volumes[i] * np.linalg.det(f[i]), 1 / n_dims),
                      np.linalg.inv(f[i]))
    return np.linalg.inv(a)


def calculate_weights(dataset: npt.NDArray, weights: npt.NDArray,
                      centroids: npt.NDArray, volumes: npt.NDArray,
                      fuzziness: float) -> npt.NDArray:
    n_points, n_clusters = weights.shape
    fcov = fuzzy_covariance_matrix(
        dataset, weights, centroids, volumes, fuzziness)
    weights = np.zeros((n_points, n_clusters))
    for i in range(n_clusters):
        for j in range(n_points):
            summ = 0
            dist_ji = mahalanobis_distance(dataset[j], centroids[i], fcov[i])
            for k in range(n_clusters):
                dist_jk = mahalanobis_distance(
                    dataset[j], centroids[k], fcov[i])
                summ += (dist_ji / dist_jk) ** (1 / (fuzziness - 1))
            weights[j, i] = 1 / summ
    return weights


def gustafson_kessel(dataset: npt.NDArray, n_clusters: int,
                     volumes: npt.NDArray = None, fuzziness: float = 2,
                     tolerance: float = 1e-4, max_iter: int = 10000
                     ) -> dict[npt.NDArray, npt.NDArray]:
    # get size of the dataset
    n_points, n_dims = dataset.shape

    if volumes == None:
        volumes = np.ones(n_clusters)

    # initialisation of weights
    weights = np.random.dirichlet(np.ones(n_clusters), size=n_points)

    # initialisation of centroids
    prev_centroids = np.zeros((n_clusters, n_dims))

    for _ in range(max_iter):
        # calculation of new centroids according to the weights of the dataset
        centroids = FCM.update_centroids(dataset, weights, fuzziness)

        # updating the weights of the dataset according to their distances to the new centroids
        weights = calculate_weights(
            dataset, weights, centroids, volumes, fuzziness)

        # check for convergence
        if euclidean_distance(prev_centroids, centroids) < tolerance:
            break
        prev_centroids = centroids
    return {"weights": weights, "centroids": centroids}
