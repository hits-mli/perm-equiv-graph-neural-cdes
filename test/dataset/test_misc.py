import os
import pickle
import hashlib
import logging
import numpy as np
import jax.numpy as jnp
import torch
import torchcde
import pytest

# Import functions from your module.
# Replace 'graph_ops' with the appropriate module name.
from dataset.misc import (
    zipf_smoothing,
    normalized_plus,
    normalized_laplacian,
    normalized_adj,
    get_graph_operator,
    padding_graph_by_time,
    get_interpolation_coeffs,
    rects_overlap,
    sample_non_overlapping_rect,
)


def test_zipf_smoothing():
    # For A = [[0,1],[1,0]], A+I becomes a matrix of ones.
    A = np.array([[0, 1], [1, 0]], dtype=float)
    result = zipf_smoothing(A)
    # Expected: each entry becomes (1/sqrt(2)) * 1 * (1/sqrt(2)) = 0.5.
    expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_normalized_plus():
    # For A = [[0,1],[1,0]], A+I becomes [[1,1],[1,1]].
    A = np.array([[0, 1], [1, 0]], dtype=float)
    result = normalized_plus(A)
    expected = np.array([[1, 1], [1, 1]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_normalized_laplacian():
    # For A = [[0,1],[1,0]] with degrees [1, 1]:
    # Expected: I - A = [[1, -1], [-1, 1]].
    A = np.array([[0, 1], [1, 0]], dtype=float)
    result = normalized_laplacian(A)
    expected = jnp.array([[1, -1], [-1, 1]], dtype=jnp.float32)
    np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-5)


def test_normalized_adj():
    # For A = [[0,1],[1,0]], degrees are 1 so the operator equals A.
    A = np.array([[0, 1], [1, 0]], dtype=float)
    result = normalized_adj(A)
    expected = np.array([[0, 1], [1, 0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_get_graph_operator():
    A = np.array([[0, 1], [1, 0]], dtype=float)
    L = np.array([[1, -1], [-1, 1]], dtype=float)
    # For "lap", expect L.
    op_lap = get_graph_operator("lap", A, L)
    np.testing.assert_allclose(op_lap, L, rtol=1e-5)
    # For "kipf", expect zipf_smoothing(A).
    op_kipf = get_graph_operator("kipf", A, L)
    expected_kipf = zipf_smoothing(A)
    np.testing.assert_allclose(op_kipf, expected_kipf, rtol=1e-5)
    # For "norm_adj", expect normalized_adj(A).
    op_norm_adj = get_graph_operator("norm_adj", A, L)
    expected_norm_adj = normalized_adj(A)
    np.testing.assert_allclose(op_norm_adj, expected_norm_adj, rtol=1e-5)
    # For any other string, expect normalized_laplacian(A).
    op_default = get_graph_operator("other", A, L)
    expected_default = normalized_laplacian(A)
    np.testing.assert_allclose(
        np.array(op_default), np.array(expected_default), rtol=1e-5
    )


def test_padding_graph_by_time_static():
    # Static graph: events_indices is None.
    # Create a dummy graph tensor.
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    t = torch.arange(3)  # time tensor with 3 time steps
    # When events_indices is None, A is repeated for each time step.
    padded = padding_graph_by_time(A, None, t, return_tensor=True)
    # Expected shape: (3, 2, 2)
    assert padded.shape == (3, 2, 2)
    for i in range(3):
        torch.testing.assert_allclose(padded[i], A)


def test_padding_graph_by_time_dynamic():
    # Dynamic graph: events_indices is provided and A_list is a list of tensors.
    A0 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    A1 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    A_list = [A0, A1]
    t = torch.arange(4)  # time steps: 0, 1, 2, 3
    events_indices = [2]
    # With padding_mode "none" and padding_last False:
    padded = padding_graph_by_time(
        A_list,
        events_indices,
        t,
        padding_mode="none",
        padding_last=False,
        return_tensor=True,
    )
    # Expected:
    # time 0: A0, time 1: NaN tensor, time 2: A1, time 3: NaN tensor.
    assert padded.shape == (4, 2, 2)
    torch.testing.assert_allclose(padded[0], A0)
    torch.testing.assert_allclose(padded[2], A1)
    assert torch.isnan(padded[1]).all()
    assert torch.isnan(padded[3]).all()

    # Now with padding_last True, last time step should use the latest graph.
    padded_last = padding_graph_by_time(
        A_list,
        events_indices,
        t,
        padding_mode="none",
        padding_last=True,
        return_tensor=True,
    )
    torch.testing.assert_allclose(padded_last[3], A1)


def test_get_interpolation_coeffs():
    # Create dummy padded_A: a tensor of shape (T, n, n); e.g., T=4, n=2.
    padded_A = torch.arange(16, dtype=torch.float32).view(4, 2, 2)
    t = torch.linspace(0, 1, steps=4)
    # Test for different interpolation methods.
    coeffs_linear = get_interpolation_coeffs(padded_A, t, interpolation="linear")
    assert isinstance(coeffs_linear, torch.Tensor)
    coeffs_cubic = get_interpolation_coeffs(padded_A, t, interpolation="cubic")
    assert isinstance(coeffs_cubic, torch.Tensor)
    coeffs_cubic_hermite = get_interpolation_coeffs(
        padded_A, t, interpolation="cubic_hermite"
    )
    assert isinstance(coeffs_cubic_hermite, torch.Tensor)
    coeffs_rectilinear = get_interpolation_coeffs(
        padded_A, t, interpolation="rectilinear"
    )
    assert isinstance(coeffs_rectilinear, torch.Tensor)


def test_rects_overlap():
    # Overlapping rectangles.
    rect1 = (0, 0, 10, 10)
    rect2 = (5, 5, 15, 15)
    assert rects_overlap(rect1, rect2) is True

    # Non-overlapping: rect1 entirely above rect2.
    rect3 = (10, 0, 20, 10)
    assert rects_overlap(rect1, rect3) is False

    # Non-overlapping: rect1 entirely to the left of rect2.
    rect4 = (0, 10, 10, 20)
    assert rects_overlap(rect1, rect4) is False


def test_sample_non_overlapping_rect_success():
    N = 10
    h = 3
    w = 3
    existing_rects = []
    rect = sample_non_overlapping_rect(N, h, w, existing_rects)
    r1, c1, r2, c2 = rect
    # Check rectangle is within bounds and has the correct dimensions.
    assert 0 <= r1 < r2 <= N
    assert 0 <= c1 < c2 <= N
    assert (r2 - r1) == h
    assert (c2 - c1) == w


def test_sample_non_overlapping_rect_failure():
    # Test that the function raises a RuntimeError if it cannot sample a non-overlapping rectangle.
    N = 5
    h = 5
    w = 5
    # Add a rectangle that covers the whole grid.
    existing_rects = [(0, 0, 5, 5)]
    with pytest.raises(RuntimeError):
        sample_non_overlapping_rect(N, h, w, existing_rects, max_attempts=10)
