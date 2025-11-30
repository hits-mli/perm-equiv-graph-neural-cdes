import os
import logging
from typing import List, Optional, Union

import numpy as np
import jax.numpy as jnp

import torch
import torchcde

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def zipf_smoothing(A: np.ndarray) -> np.ndarray:
    """
    Apply Zipf smoothing to the adjacency matrix A.

    Args:
        A (np.ndarray): The adjacency matrix.
    Returns:
        np.ndarray: The smoothed adjacency matrix.
    """
    A_prime = A + np.eye(A.shape[0])
    out_degree = np.array(A_prime.sum(1), dtype=np.float32)
    int_degree = np.array(A_prime.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A_prime @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_plus(A: np.ndarray) -> np.ndarray:
    """
    Apply normalized plus operation to the adjacency matrix A.

    Args:
        A (np.ndarray): The adjacency matrix.

    Returns:
        np.ndarray: The normalized plus matrix.
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = (
        np.diag(out_degree_sqrt_inv)
        @ (A + np.eye(A.shape[0]))
        @ np.diag(int_degree_sqrt_inv)
    )
    return mx_operator


def normalized_laplacian(A: np.ndarray) -> jnp.ndarray:
    """
    Compute the normalized Laplacian of the adjacency matrix A.

    Args:
        A (np.ndarray): The adjacency matrix.

    Returns:
        jnp.ndarray: The normalized Laplacian matrix.
    """
    A = A + jnp.eye(A.shape[0])  # Add self-loops

    out_degree = jnp.array(A.sum(1), dtype=np.float32)
    int_degree = jnp.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = jnp.power(out_degree, -0.5)
    int_degree_sqrt_inv = jnp.power(int_degree, -0.5)
    mx_operator = jnp.eye(A.shape[0]) - jnp.diag(out_degree_sqrt_inv) @ A @ jnp.diag(
        int_degree_sqrt_inv
    )
    return mx_operator


def normalized_adj(A: np.ndarray) -> np.ndarray:
    """
    Compute the normalized adjacency matrix of A.

    Args:
        A (np.ndarray): The adjacency matrix.

    Returns:
        np.ndarray: The normalized adjacency matrix.
    """
    A = A + jnp.eye(A.shape[0])  # Add self-loops

    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def get_graph_operator(operator_type, A, L):
    operator_type = operator_type.lower()
    if operator_type == "lap":
        OM = L
    elif operator_type == "kipf":
        OM = zipf_smoothing(A)
    elif operator_type == "norm_adj":
        OM = normalized_adj(A)
    else:
        OM = normalized_laplacian(A)  # L # normalized_adj

    return OM


def padding_graph_by_time(
    A_list,
    events_indices,
    t,
    padding_mode="none",
    padding_last=False,
    return_tensor=True,
):
    """
    :param A_list:
    :param events_indices:
    :param t:
    :param padding_mode:
    :param padding_last: if True, we assume the graph structure is not changed since last observation
    :param return_tensor:
    :return:
    """
    # static graph
    if events_indices is None:
        if return_tensor:
            return A_list.unsqueeze(0).repeat([t.size(0), 1, 1])
        else:
            return [A_list for _ in range(t.size(0))]

    # dynamic graph
    padded_A_list = [A_list[0]]
    event_idx = 0
    for idx in range(1, len(t)):
        if event_idx < len(events_indices) and idx == events_indices[event_idx]:
            event_idx += 1
            padded_A_list.append(A_list[event_idx])
        else:
            if padding_mode == "none":
                padded_A_list.append(torch.ones_like(A_list[0]) * torch.nan)
            else:
                padded_A_list.append(A_list[event_idx])

    if padding_last and events_indices[-1] != len(t) - 1:
        padded_A_list[-1] = A_list[event_idx]

    if return_tensor:
        return torch.stack(padded_A_list, dim=0)

    return padded_A_list


def get_interpolation_coeffs(padded_A, t, interpolation="linear"):
    # contiguous()
    padded_A = padded_A.view(padded_A.size(0), -1).t()
    t_index = t.unsqueeze(0).repeat(padded_A.size(0), 1)
    X = torch.stack([t_index, padded_A], dim=-1)

    if interpolation == "cubic":
        coeffs = torchcde.natural_cubic_coeffs(X)
    elif interpolation == "cubic_hermite":
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
    elif interpolation == "rectilinear":
        coeffs = torchcde.linear_interpolation_coeffs(X, rectilinear=0)
    else:
        coeffs = torchcde.linear_interpolation_coeffs(X)

    return coeffs


# Create a batched grid of zeros.
# --- Helper functions ---
def rects_overlap(rect1, rect2):
    """
    Check if two rectangles overlap.
    Rectangles are defined as (r1, c1, r2, c2) where r2 and c2 are exclusive.
    """
    return not (
        rect1[2] <= rect2[0]  # rect1 is entirely above rect2
        or rect1[0] >= rect2[2]  # rect1 is entirely below rect2
        or rect1[3] <= rect2[1]  # rect1 is entirely left of rect2
        or rect1[1] >= rect2[3]  # rect1 is entirely right of rect2
    )


def sample_non_overlapping_rect(N, h, w, existing_rects, max_attempts=100):
    """
    Sample a random top-left location for a rectangle of height h and width w
    so that the rectangle lies within a grid of size (N x N) and does not overlap
    any rectangle in existing_rects.

    Returns:
        rect (tuple): (r1, c1, r2, c2) with r2, c2 exclusive.
    """
    for _ in range(max_attempts):
        r = np.random.randint(0, N - h + 1)
        c = np.random.randint(0, N - w + 1)
        rect = (r, c, r + h, c + w)
        if not any(rects_overlap(rect, ex) for ex in existing_rects):
            return rect
    raise RuntimeError(
        "Could not sample a non-overlapping rectangle after {} attempts".format(
            max_attempts
        )
    )
