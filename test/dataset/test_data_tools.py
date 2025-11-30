import math
import numpy as np
import pytest
import networkx as nx
import scipy.sparse as sp

# Import the functions from your module.
# Replace 'graph_utils' with the actual module name if different.
from dataset.data_tools import (
    grid_8_neighbor_graph,
    generate_node_mapping,
    networkx_reorder_nodes,
    gen_events_happen_time,
    gen_events_happen_graph,
)


def test_grid_8_neighbor_graph():
    N = 3
    A = grid_8_neighbor_graph(N)
    expected_shape = (N * N, N * N)
    assert A.shape == expected_shape, f"Expected shape {expected_shape}, got {A.shape}"

    # Test an interior (center) node: for a 3x3 grid, node at (1,1) is index 4.
    center_neighbors = A[4].sum()
    assert (
        center_neighbors == 8
    ), f"Expected 8 neighbors for center node, got {center_neighbors}"

    # Test a corner node: for example, node at (0,0) is index 0 (neighbors: (0,1), (1,0), (1,1))
    corner_neighbors = A[0].sum()
    assert (
        corner_neighbors == 3
    ), f"Expected 3 neighbors for corner node, got {corner_neighbors}"


def test_generate_node_mapping_degree():
    # Create a star graph: center node 0 has highest degree.
    G = nx.star_graph(4)
    mapping = generate_node_mapping(G, type="degree")
    assert mapping is not None, "Mapping should not be None for type 'degree'."

    # In a star graph, the center (node 0) should have highest degree and be mapped to 0.
    assert mapping[0] == 0, f"Expected node 0 to map to 0, got {mapping[0]}"
    # Ensure all nodes in G are present in the mapping.
    assert set(mapping.keys()) == set(
        G.nodes()
    ), "Mapping keys do not match graph nodes."


def test_generate_node_mapping_community():
    # Use a graph with known community structure.
    G = nx.karate_club_graph()
    mapping = generate_node_mapping(G, type="community")
    assert mapping is not None, "Mapping should not be None for type 'community'."
    # The mapping should cover all nodes.
    assert set(mapping.keys()) == set(
        G.nodes()
    ), "Mapping keys do not match graph nodes."


def test_generate_node_mapping_none():
    # When type is None, the mapping should be None.
    G = nx.path_graph(5)
    mapping = generate_node_mapping(G, type=None)
    assert mapping is None, "Expected mapping to be None when type is None."


def test_networkx_reorder_nodes_degree():
    G = nx.star_graph(4)
    G_reordered = networkx_reorder_nodes(G, type="degree")
    # The reordered graph should have nodes labeled 0 to n-1.
    expected_nodes = set(range(len(G.nodes())))
    assert (
        set(G_reordered.nodes()) == expected_nodes
    ), "Reordered nodes do not match expected labels."

    # The reordered graph should be isomorphic to the original.
    assert nx.is_isomorphic(
        G, G_reordered
    ), "Reordered graph is not isomorphic to the original."


def test_networkx_reorder_nodes_none():
    G = nx.path_graph(5)
    G_reordered = networkx_reorder_nodes(G, type=None)
    # With no mapping, the original graph is returned.
    assert set(G_reordered.nodes()) == set(
        G.nodes()
    ), "Graph should remain unchanged when type is None."


def test_gen_events_happen_time_dynamic():
    # Set a fixed seed for reproducibility.
    np.random.seed(42)
    batch_size = 2
    num_t = 10
    # Create a dummy time array of shape (batch_size, num_t)
    t = np.tile(np.arange(num_t), (batch_size, 1))
    event_times = 3
    split_ratio = [0.8, 0.2]

    event_ts, event_indices = gen_events_happen_time(
        t, event_times, split_ratio, enable_all_dynamic=True
    )

    # event_ts should have shape (batch_size, event_times)
    assert event_ts.shape == (
        batch_size,
        event_times,
    ), f"Expected event_ts shape {(batch_size, event_times)}, got {event_ts.shape}"
    # event_indices should be a 1D array of length event_times
    assert (
        len(event_indices) == event_times
    ), f"Expected event_indices length {event_times}, got {len(event_indices)}"
    # Verify that the event indices are in sorted order.
    assert np.all(
        np.diff(event_indices) >= 0
    ), "Event indices should be sorted in ascending order."


def test_gen_events_happen_time_non_dynamic_error():
    # For the non-dynamic branch, note that event_ts is never appended to,
    # so stacking an empty list should raise an error.
    np.random.seed(42)
    batch_size = 1
    num_t = 10
    t = np.tile(np.arange(num_t), (batch_size, 1))
    event_times = 3
    split_ratio = [0.8, 0.2]

    with pytest.raises(ValueError):
        # Expecting a ValueError from np.stack when event_ts is empty.
        gen_events_happen_time(t, event_times, split_ratio, enable_all_dynamic=False)


def test_gen_events_happen_graph():
    np.random.seed(42)
    # Create a simple batched adjacency matrix for a graph.
    # For instance, a graph with one batch element and 3 nodes.
    A = np.array(
        [[[0, 1, 0], [1, 0, 1], [0, 1, 0]]],
        dtype=float,
    )
    event_times = 2
    A_list, D_list, L_list = gen_events_happen_graph(A, event_times, p=0.005)

    # The lists should contain event_times + 1 elements (including the initial state).
    assert (
        len(A_list) == event_times + 1
    ), f"Expected {event_times + 1} adjacency matrices, got {len(A_list)}"
    assert (
        len(D_list) == event_times + 1
    ), f"Expected {event_times + 1} degree matrices, got {len(D_list)}"
    assert (
        len(L_list) == event_times + 1
    ), f"Expected {event_times + 1} Laplacian matrices, got {len(L_list)}"

    # Each matrix should have the same shape as the original.
    for mat in A_list:
        assert mat.shape == A.shape, "Adjacency matrix shape changed during events."

    # The first adjacency matrix in the list should be identical to the original.
    np.testing.assert_array_equal(A_list[0], A)
