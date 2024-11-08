import numpy as np
import numpy.typing as npt

def partition_coefficient(weights: npt.NDArray, fuzziness: float = 2) -> npt.NDArray:
    """Partition Coefficient Index: Measures the degree of overlap between clusters.
    Higher values are better; means less overlap between clusters.

    Args:
        weights (npt.NDArray): _description_

    Returns:
        npt.NDArray: _description_
    """
    return np.sum(np.power(weights, fuzziness)) / weights.shape[0]