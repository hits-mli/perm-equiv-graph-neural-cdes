import torch
import torch.nn as nn
import jax.numpy as jnp


class SIRDynamics(nn.Module):
    def __init__(self, A, beta, gamma):
        super(SIRDynamics, self).__init__()
        self.A = A  # Adjacency matrix
        self.beta = beta
        self.gamma = gamma

    def forward(self, t, x):
        """
        Compute time-derivative for the networked SIR model.

        Args:
            t:   Time scalar (required by odeint interface, but unused here)
            x:   State vector of shape (N,3), representing of [S, I, R]

        Returns:
            dydt: Derivative (dS, dI, dR), shape (N,3)
        """
        S, I, R = x[:, 0], x[:, 1], x[:, 2]
        # Infection flux from neighbors: β * S_i * sum_j A_ij * I_j
        infection = self.beta * S * (self.A @ I)
        dS = -infection
        dI = infection - self.gamma * I
        dR = self.gamma * I

        return jnp.stack([dS, dI, dR], axis=1)
