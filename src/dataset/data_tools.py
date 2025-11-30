import math
import numpy as np
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms import community


def grid_8_neighbor_graph(N):
    """
    Build a discrete grid graph where each node has 8 neighbors.

    :param N:  The grid side length (i.e. sqrt of the number of nodes).
    :return: A, the adjacency matrix as a NumPy array of floats.
    """
    N = int(N)
    n = int(N**2)
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    A = np.zeros((n, n), dtype=float)
    for x in range(N):
        for y in range(N):
            index = x * N + y
            for i in range(len(dx)):
                newx = x + dx[i]
                newy = y + dy[i]
                if 0 <= newx < N and 0 <= newy < N:
                    index2 = newx * N + newy
                    A[index, index2] = 1.0
    return A


def generate_node_mapping(G, type=None):
    """
    Generate a mapping for nodes based on the chosen criterion.

    :param G: A NetworkX graph.
    :param type: The type of reordering ("degree" or "community"). If None, no mapping is used.
    :return: A dictionary mapping original node IDs to new node IDs.
    """
    if type == "degree":
        s = sorted(G.degree, key=lambda x: x[1], reverse=True)
        new_map = {s[i][0]: i for i in range(len(s))}
    elif type == "community":
        cs = list(community.greedy_modularity_communities(G))
        l = []
        for c in cs:
            l += list(c)
        new_map = {l[i]: i for i in range(len(l))}
    else:
        new_map = None

    return new_map


def networkx_reorder_nodes(G, type=None):
    """
    Reorder the nodes of a NetworkX graph based on the mapping generated from `generate_node_mapping`.

    :param G: A NetworkX graph.
    :param type: The type of reordering ("degree", "community", or None).
    :return: A reordered NetworkX graph.
    """
    nodes_map = generate_node_mapping(G, type)
    if nodes_map is None:
        return G
    # Get the sparse (COO) representation of the graph.
    C = nx.to_scipy_sparse_array(G, format="coo")
    new_row = np.array([nodes_map[x] for x in C.row], dtype=np.int32)
    new_col = np.array([nodes_map[x] for x in C.col], dtype=np.int32)
    new_C = sp.coo_matrix((C.data, (new_row, new_col)), shape=C.shape)
    new_G = nx.from_scipy_sparse_array(new_C)
    return new_G


def gen_events_happen_time(t, event_times, split_ratio, enable_all_dynamic=False):
    """
    Randomly choose time indices (and corresponding times) at which graph events occur.

    :param t: A NumPy array of time stamps.
    :param event_times: The number of events to sample.
    :param split_ratio: A tuple/list with ratios for splitting (e.g., training vs. testing).
    :param enable_all_dynamic: If True, sample events for both training and testing segments.
    :return: (event_t, event_indices) where event_t are the times at which events occur.
    """
    batch_size, num_t = t.shape
    event_ts = []
    event_indices_list = []
    if not enable_all_dynamic:
        n_train = int(num_t * split_ratio[0])
        random_indices = np.random.permutation(n_train - 2) + 2
        event_indices = random_indices[:event_times]
    else:
        train_event_times = math.ceil(event_times * split_ratio[0])
        test_event_time = event_times - train_event_times
        n_train = int(num_t * split_ratio[0])
        for i in range(batch_size):
            train_random_indices = np.random.permutation(n_train - 2) + 2
            test_random_indices = np.random.permutation(num_t - n_train) + n_train
            event_indices = np.concatenate(
                (
                    train_random_indices[:train_event_times],
                    test_random_indices[:test_event_time],
                )
            )
            event_indices = np.sort(event_indices)
            event_indices_list.append(event_indices)
            event_ts.append(t[i, event_indices])
    return np.stack(event_ts, axis=0), event_indices


def gen_events_happen_graph(A, event_times, p=0.1):
    """
    Generate a list of adjacency matrices (and corresponding degree and Laplacian matrices)
    that simulate events happening on the graph (dropping/adding edges stochastically).

    :param A: The initial adjacency matrix as a NumPy array.
    :param event_times: Number of events to simulate.
    :param p: The base probability of an event happening on each edge.
    :return: (A_list, D_list, L_list) lists for each event.
    """

    batch_size, n, _ = A.shape

    # Compute the initial degree and Laplacian matrices for each batch element.
    D = np.array([np.diag(A[i].sum(axis=1)) for i in range(batch_size)])
    L = D - A

    A_list = [A.copy()]
    D_list = [D.copy()]
    L_list = [L.copy()]

    # For each event, modify the graph structure for each batch element.
    for _ in range(event_times):
        # Work on a copy of the batched adjacency matrix.
        A_new = A.copy()

        # Determine which edges to drop. Use a higher drop probability.
        drop_prob = 20 * p
        drop = np.random.binomial(1, drop_prob, size=A.shape).astype(bool)
        A_new[drop] = 0.0

        # Determine which edges to add.
        add = np.random.binomial(1, p, size=A.shape).astype(bool)
        A_new[add] = 1.0

        # Update the degree and Laplacian matrices for each batch element.
        D_new = np.array([np.diag(A_new[i].sum(axis=1)) for i in range(batch_size)])
        L_new = D_new - A_new

        # Append the new matrices to the lists.
        A_list.append(A_new.copy())
        D_list.append(D_new.copy())
        L_list.append(L_new.copy())

        # Update A for the next event.
        A = A_new

    return A_list, D_list, L_list
