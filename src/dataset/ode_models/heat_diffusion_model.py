import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO  # For sparse matrices


class HeatDiffusion:
    """
    Heat diffusion ODE: dX/dt = -k * L * X
    """

    def __init__(self, L, k=1.0):
        """
        :param L: Graph Laplacian (dense or sparse JAX array)
        :param k: Heat diffusion coefficient
        """
        self.k = k
        self.L = (
            L if isinstance(L, BCOO) else jnp.array(L)
        )  # Convert to JAX array if needed

    def __call__(self, t, x):
        """
        Compute dX/dt given state x and time t.
        :param t: Time step (not used for autonomous systems)
        :param x: State (n x dim matrix)
        :return: Updated derivative dX/dt
        """
        if isinstance(self.L, BCOO):  # Sparse matrix multiplication
            f = self.L @ x
        else:  # Dense multiplication
            f = jnp.dot(self.L, x)

        return -self.k * f  # Apply heat diffusion
