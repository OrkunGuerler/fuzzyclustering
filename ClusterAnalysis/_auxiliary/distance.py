import numpy as np
import numpy.typing as npt


def sqrd_euclidean_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Squared Euclidean Distance Function

    Args:
        x (npt.NDArray): Initial point
        y (npt.NDArray): Final point

    Returns:
        npt.NDArray: _description_
    """
    return np.sum(np.power(x - y, 2))


def pairwise_sqrd_euclidean_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Pairwise Squared Euclidean Distance Function

    Args:
        x (npt.NDArray): Initial point
        y (npt.NDArray): Final point

    Returns:
        npt.NDArray: _description_
    """
    return np.sum(np.power(x[:, np.newaxis] - y, 2), axis=2)


def euclidean_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Euclidean Distance Function

    Args:
        x (npt.NDArray): Initial point
        y (npt.NDArray): Final point

    Returns:
        npt.NDArray: _description_
    """
    return np.sqrt(sqrd_euclidean_distance(x, y))


def pairwise_euclidean_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Pairwise Euclidean Distance Function

    Args:
        x (npt.NDArray): Initial point
        y (npt.NDArray): Final point

    Returns:
        npt.NDArray: _description_
    """
    return np.sqrt(pairwise_sqrd_euclidean_distance(x, y))


# -------------------------------------------------------------------------------

def manhattan_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Manhattan Distance Function

    Args:
        x (npt.NDArray): Initial point
        y (npt.NDArray): Final point

    Returns:
        npt.NDArray: _description_
    """
    return np.sum(np.abs(x - y))


# -------------------------------------------------------------------------------

def mahalanobis_distance(x: npt.NDArray, y: npt.NDArray, c: npt.NDArray) -> npt.NDArray:
    """Mahalanobis Distance Function

    Args:
        x (npt.NDArray): Initial point
        y (npt.NDArray): Final point
        c (npt.NDArray): Covariance matrix

    Returns:
        npt.NDArray: _description_
    """
    inv_cov = np.linalg.inv(c)
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))


# -------------------------------------------------------------------------------


def rbf_kernel_distance(x: npt.NDArray, y: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """RBF(Gaussian) Kernel Distance Function

    Args:
        x (npt.NDArray): Initial point
        y (npt.NDArray): Final point
        b (npt.NDArray): Bandwidth of kernel

    Returns:
        npt.NDArray: _description_
    """
    return np.exp(- sqrd_euclidean_distance(x, y) / (b ** 2))
