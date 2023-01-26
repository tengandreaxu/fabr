import numpy as np


def get_block_sizes(
    number_random_features: int, small_subset_size: int, voc_grid: list
) -> list:
    """returns a list of block sizes"""
    block_sizes = (
        np.arange(0, number_random_features, small_subset_size).astype(int).tolist()
    )
    if number_random_features not in block_sizes:
        block_sizes += [number_random_features]

    # if grid point in voc_grid is not in block_sizes, we add them in the block_sizes
    block_sizes = list(set(block_sizes + voc_grid))

    block_sizes.sort()  # sort grid points
    return block_sizes
