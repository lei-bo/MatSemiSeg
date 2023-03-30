import numpy as np
import itertools


def brute_force_search(sim_matrix, k, id_choices=None):
    best_sim = np.NINF
    selection = tuple()
    n = sim_matrix.shape[0]
    indices = set(np.arange(n))
    for subset in itertools.combinations(indices, k):
        if id_choices and not set(subset).issubset(set(id_choices)):
            continue
        remainder = tuple(indices.difference(subset))
        sim = sim_matrix[subset, :][:, remainder].max(axis=0).sum()
        if sim > best_sim:
            best_sim = sim
            selection = subset
    return np.array(selection)


if __name__ == '__main__':
    np.random.seed(0)
    import time
    for n, k in itertools.product([10, 20, 50], [1, 2, 5]):
        print(n, k)
        sim_mat = np.random.rand(n, n)
        for i in range(sim_mat.shape[0]):
            sim_mat[i, i] = 1
            for j in range(i, sim_mat.shape[1]):
                sim_mat[i, j] = sim_mat[j, i]
        start = time.time()
        res_bf = brute_force_search(sim_mat, k)
        print(f"complete brute force search in {time.time() - start:.3f} s")
