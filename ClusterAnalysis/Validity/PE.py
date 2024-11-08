import numpy as np
import numpy.typing as npt

def partition_entropy(weights: npt.NDArray, base: float = np.e) -> npt.NDArray:
    """Partition Entropy Index: Quantifies the fuzziness or uncertainty in the clustering. It is calculated using the entropy of the membership values.
    Lower values are better; means less fuzziness and clearer distinctions between clusters.

    Args:
        weights (npt.NDArray): _description_

    Returns:
        npt.NDArray: _description_
    """
    return - np.sum(np.multiply(weights, np.emath.logn(base, weights))) / weights.shape[0]