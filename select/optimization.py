import numpy as np
import cvxpy as cp
import itertools


def milp_optimize(sim_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Optimize a mixed-integer linear program to find a subset of size k such that
    the sum of maximum similarity between the subset and the rest of the set is
    maximized.
    :param sim_matrix: entry [i,j] gives the similarity between data
    point i and j
    :param k: size of the subset
    :return: indices of the data in the subset
    """
    n = sim_matrix.shape[0]
    # if the data point is in the subset
    x = cp.Variable(n, boolean=True)
    # auxiliary variable
    z = cp.Variable(n)
    # b[i,j] == 1 if data j is in the subset and s[i,j] is maximum
    b = cp.Variable((n, n), boolean=True)

    constraints = [cp.sum(x) == k,
                   cp.sum(b, axis=1) == np.ones(n)]
    for j in range(n):
        constraints.append(z >= cp.multiply(sim_matrix[:, j], x[j]))
        constraints.append(z <= cp.multiply(sim_matrix[:, j], x[j]) + 2*(1-b[:, j]))

    obj = cp.Maximize(cp.sum(z))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.GUROBI)
    return np.where(x.value == 1)[0]


def _brute_force_search(sim_matrix, k):
    best_sim = 0
    selection = tuple()
    n = sim_matrix.shape[0]
    indices = set(np.arange(n))
    for subset in itertools.combinations(indices, k):
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
        res_milp = milp_optimize(sim_mat, k)
        print(f"complete MILP optimization in {time.time() - start:.3f} s")
        start = time.time()
        res_bf = _brute_force_search(sim_mat, k)
        print(f"complete brute force search in {time.time() - start:.3f} s")
        assert np.all(res_milp == res_bf)
