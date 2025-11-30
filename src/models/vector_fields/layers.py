import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array
import equinox.nn as nn
import math


class ConvLayer(eqx.Module):
    """A graph convolutional layer with linear transformation and RMS normalization.

    Attributes:
        linear (eqx.nn.Linear): Linear transformation layer.
        norm (eqx.nn.RMSNorm): RMS normalization layer.
    """

    linear: nn.Linear
    norm: nn.RMSNorm

    def __init__(self, input_dim: int, output_dim: int, *, key: jr.PRNGKey, **kwargs):
        """Initializes the ConvLayer.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            key (jr.PRNGKey): Random key for initializing parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        skey, nkey = jr.split(key, 2)
        self.linear = eqx.nn.Linear(input_dim, output_dim, key=skey)
        self.norm = nn.RMSNorm(input_dim)

    def __call__(self, node_feats: Array, adj_matrix: Array) -> Array:
        """Forward pass of the convolutional layer.

        Args:
            node_feats (Array): Node features.
            adj_matrix (Array): Adjacency matrix.
        Returns:
            Array: Output features after convolution.
        """
        node_feats = jax.vmap(self.norm)(node_feats)
        m = jax.vmap(self.linear)(node_feats)
        m = m + adj_matrix @ m
        return m


class ConvEquivFusionLayer(eqx.Module):
    """A convolutional layer with linear equivariant fusion of adjacency matrix and control signal.

    Attributes:
        param1 (Array): First fusion parameter.
        param2 (Array): Second fusion parameter.
        param3 (Array): Third fusion parameter.
        param4 (Array): Fourth fusion parameter.
        param5 (Array): Fifth fusion parameter.
        param6 (Array): Sixth fusion parameter.
        param7 (Array): Seventh fusion parameter.
        param8 (Array): Eighth fusion parameter.
        conv_layer (ConvLayer): Convolutional layer.
    """

    param1: Array
    param2: Array
    param3: Array
    param4: Array
    param5: Array
    param6: Array
    param7: Array
    param8: Array
    conv_layer: ConvLayer

    def __init__(self, input_dim: int, output_dim: int, *, key: jr.PRNGKey):
        """Initializes the ConvEquivFusionLayer.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            key (jr.PRNGKey): Random key for initializing parameters.
        """
        super(ConvEquivFusionLayer, self).__init__()

        lim = 1
        p1key, p2key, p3key, p4key, p5key, p6key, p7key, p8key, key = jr.split(key, 9)
        self.param1 = 1 / 15 * jr.uniform(p1key, (2,), minval=-lim, maxval=lim)
        self.param2 = 1 / 15 * jr.uniform(p2key, (2,), minval=-lim, maxval=lim)
        self.param3 = 1 / 15 * jr.uniform(p3key, (2,), minval=-lim, maxval=lim)
        self.param4 = 1 / 15 * jr.uniform(p4key, (2,), minval=-lim, maxval=lim)
        self.param5 = 1 / 15 * jr.uniform(p5key, (2,), minval=-lim, maxval=lim)
        self.param6 = 1 / 15 * jr.uniform(p6key, (2,), minval=-lim, maxval=lim)
        self.param7 = 1 / 15 * jr.uniform(p7key, (2,), minval=-lim, maxval=lim)
        self.param8 = 1 / 15 * jr.uniform(p8key, (2,), minval=-lim, maxval=lim)

        conv_key, key = jr.split(key, 2)
        self.conv_layer = ConvLayer(
            input_dim=input_dim, output_dim=output_dim, key=conv_key
        )

    def _fusion(self, adjacency: Array, control_gradient: Array) -> Array:
        """Fuses the adjacency matrix and control gradient as a linear combination of the basis terms.

        Args:
            adjacency (Array): Adjacency matrix.
            control_gradient (Array): Control gradient matrix.

        Returns:
            Array: Fused adjacency matrix.
        """
        n = adjacency.shape[0]

        term_1 = (1.0 + self.param1[0]) * adjacency + (
            1.0 + self.param1[1]
        ) * control_gradient

        term_2 = self.param2[0] * jnp.transpose(adjacency) + self.param2[
            1
        ] * jnp.transpose(control_gradient)

        term_3 = self.param3[0] * jnp.diag(jnp.diag(adjacency)) + self.param3[
            1
        ] * jnp.diag(jnp.diag(control_gradient))

        # w3 - sum rows on rows
        term_4 = self.param4[0] / n * jnp.transpose(
            jnp.tile(jnp.sum(adjacency, axis=1), (n, 1))
        ) + self.param4[1] / n * jnp.transpose(
            jnp.tile(jnp.sum(control_gradient, axis=1), (n, 1))
        )

        # w4 - sum rows on cols
        term_5 = self.param5[0] / n * jnp.tile(
            jnp.sum(adjacency, axis=1), (n, 1)
        ) + self.param5[1] / n * jnp.tile(jnp.sum(control_gradient, axis=1), (n, 1))

        # w5 - sum rows on diag
        term_6 = self.param6[0] / n * jnp.diag(
            jnp.sum(adjacency, axis=1)
        ) + self.param6[1] / n * jnp.diag(jnp.sum(control_gradient, axis=1))

        # w9 - sum all on all
        term_7 = self.param7[0] / n**2 * jnp.full(
            adjacency.shape, jnp.sum(adjacency)
        ) + self.param7[1] / n**2 * jnp.full(
            control_gradient.shape, jnp.sum(adjacency)
        )

        # w10 - sum all on diag
        term_8 = (
            (
                self.param8[0] * jnp.sum(adjacency)
                + self.param8[1] * jnp.sum(control_gradient)
            )
            / n**2
            * jnp.eye(n)
        )

        return term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8

    def __call__(
        self, node_feats: Array, adj_matrix: Array, control_gradient: Array
    ) -> Array:
        """Forward pass of the convolutional equivariant fusion layer.

        Args:
            node_feats (Array): Node features.
            adj_matrix (Array): Adjacency matrix.
            control_gradient (Array): Control gradient matrix.
        Returns:
            Array: Output features after convolution.
        """
        fused_adjacency = self._fusion(adj_matrix, control_gradient)

        out = self.conv_layer(node_feats, fused_adjacency)
        return out


class ConvEquivFusionDirectedLayer(eqx.Module):
    """A convolutional layer with linear equivariant fusion of adjacency matrix and control signal.

    Attributes:
        param1 (Array): First fusion parameter.
        param2 (Array): Second fusion parameter.
        param3 (Array): Third fusion parameter.
        param4 (Array): Fourth fusion parameter.
        param5 (Array): Fifth fusion parameter.
        param6 (Array): Sixth fusion parameter.
        param7 (Array): Seventh fusion parameter.
        param8 (Array): Eighth fusion parameter.
        conv_layer (ConvLayer): Convolutional layer.
    """

    param1: Array
    param2: Array
    param3: Array
    param4: Array
    param4_prime: Array
    param5: Array
    param5_prime: Array
    param6: Array
    param6_prime: Array
    param7: Array
    param8: Array
    conv_layer: ConvLayer

    def __init__(self, input_dim: int, output_dim: int, *, key: jr.PRNGKey):
        """Initializes the ConvEquivFusionLayer.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            key (jr.PRNGKey): Random key for initializing parameters.
        """
        super(ConvEquivFusionDirectedLayer, self).__init__()

        lim = 1
        (
            p1key,
            p2key,
            p3key,
            p4key,
            p4_primekey,
            p5key,
            p5_primekey,
            p6key,
            p6_primekey,
            p7key,
            p8key,
            key,
        ) = jr.split(key, 12)
        self.param1 = 1 / 15 * jr.uniform(p1key, (2,), minval=-lim, maxval=lim)
        self.param2 = 1 / 15 * jr.uniform(p2key, (2,), minval=-lim, maxval=lim)
        self.param3 = 1 / 15 * jr.uniform(p3key, (2,), minval=-lim, maxval=lim)
        self.param4 = 1 / 15 * jr.uniform(p4key, (2,), minval=-lim, maxval=lim)
        self.param4_prime = (
            1 / 15 * jr.uniform(p4_primekey, (2,), minval=-lim, maxval=lim)
        )
        self.param5 = 1 / 15 * jr.uniform(p5key, (2,), minval=-lim, maxval=lim)
        self.param5_prime = (
            1 / 15 * jr.uniform(p5_primekey, (2,), minval=-lim, maxval=lim)
        )
        self.param6 = 1 / 15 * jr.uniform(p6key, (2,), minval=-lim, maxval=lim)
        self.param6_prime = (
            1 / 15 * jr.uniform(p5_primekey, (2,), minval=-lim, maxval=lim)
        )
        self.param7 = 1 / 15 * jr.uniform(p7key, (2,), minval=-lim, maxval=lim)
        self.param8 = 1 / 15 * jr.uniform(p8key, (2,), minval=-lim, maxval=lim)

        conv_key, key = jr.split(key, 2)
        self.conv_layer = ConvLayer(
            input_dim=input_dim, output_dim=output_dim, key=conv_key
        )

    def _fusion(self, adjacency: Array, control_gradient: Array) -> Array:
        """Fuses the adjacency matrix and control gradient as a linear combination of the basis terms.

        Args:
            adjacency (Array): Adjacency matrix.
            control_gradient (Array): Control gradient matrix.

        Returns:
            Array: Fused adjacency matrix.
        """
        n = adjacency.shape[0]

        term_1 = (1.0 + self.param1[0]) * adjacency + (
            1.0 + self.param1[1]
        ) * control_gradient

        term_2 = self.param2[0] * jnp.transpose(adjacency) + self.param2[
            1
        ] * jnp.transpose(control_gradient)

        term_3 = self.param3[0] * jnp.diag(jnp.diag(adjacency)) + self.param3[
            1
        ] * jnp.diag(jnp.diag(control_gradient))

        # w3 - sum rows on rows
        term_4 = self.param4[0] / n * jnp.transpose(
            jnp.tile(jnp.sum(adjacency, axis=0), (n, 1))
        ) + self.param4[1] / n * jnp.transpose(
            jnp.tile(jnp.sum(control_gradient, axis=0), (n, 1))
        )

        # w3_prime - sum cols on rows
        term_4_prime = self.param4_prime[0] / n * jnp.tile(
            jnp.sum(adjacency, axis=1), (n, 1)
        ) + self.param4_prime[1] / n * jnp.tile(
            jnp.sum(control_gradient, axis=0), (n, 1)
        )

        # w4 - sum rows on cols
        term_5 = self.param5[0] / n * jnp.tile(
            jnp.sum(adjacency, axis=0), (n, 1)
        ) + self.param5[1] / n * jnp.tile(jnp.sum(control_gradient, axis=0), (n, 1))

        # w4_prime - sum cols on cols
        term_5_prime = self.param5_prime[0] / n * jnp.tile(
            jnp.sum(adjacency, axis=1), (n, 1)
        ) + self.param5_prime[1] / n * jnp.tile(
            jnp.sum(control_gradient, axis=1), (n, 1)
        )

        # w5 - sum rows on diag
        term_6 = self.param6[0] / n * jnp.diag(
            jnp.sum(adjacency, axis=0)
        ) + self.param6[1] / n * jnp.diag(jnp.sum(control_gradient, axis=0))

        # w5_prime - sum cols on diag
        term_6_prime = self.param6_prime[0] / n * jnp.diag(
            jnp.sum(adjacency, axis=1)
        ) + self.param6_prime[1] / n * jnp.diag(jnp.sum(control_gradient, axis=1))

        # w9 - sum all on all
        term_7 = self.param7[0] / n**2 * jnp.full(
            adjacency.shape, jnp.sum(adjacency)
        ) + self.param7[1] / n**2 * jnp.full(
            control_gradient.shape, jnp.sum(adjacency)
        )

        # w10 - sum all on diag
        term_8 = (
            (
                self.param8[0] * jnp.sum(adjacency)
                + self.param8[1] * jnp.sum(control_gradient)
            )
            / n**2
            * jnp.eye(n)
        )

        return (
            term_1
            + term_2
            + term_3
            + term_4
            + term_4_prime
            + term_5
            + term_5_prime
            + term_6
            + term_6_prime
            + term_7
            + term_8
        )

    def __call__(
        self, node_feats: Array, adj_matrix: Array, control_gradient: Array
    ) -> Array:
        """Forward pass of the convolutional equivariant fusion layer.

        Args:
            node_feats (Array): Node features.
            adj_matrix (Array): Adjacency matrix.
            control_gradient (Array): Control gradient matrix.
        Returns:
            Array: Output features after convolution.
        """
        fused_adjacency = self._fusion(adj_matrix, control_gradient)

        out = self.conv_layer(node_feats, fused_adjacency)
        return out


class ConvPreMultFusionLayer(eqx.Module):
    """A convolutional layer with full fusion, that premultiplies both the adjacency matrix and the control signal by a fusion matrix.

    Attributes:
        fusion_1 (Array): First fusion matrix.
        fusion_2 (Array): Second fusion matrix.
        conv_layer (ConvLayer): Convolutional layer.
    """

    fusion_1: Array
    fusion_2: Array
    conv_layer: ConvLayer

    def __init__(self, input_dim: int, output_dim: int, *, key: jr.PRNGKey):
        """Initializes the ConvFullFusionLayer.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            key (jr.PRNGKey): Random key for initializing parameters.
        """
        super(ConvPreMultFusionLayer, self).__init__()

        lim = 1
        fkey1, fkey2, key = jr.split(key, 3)
        self.fusion_1 = jr.uniform(fkey1, (400, 400))
        self.fusion_2 = jr.uniform(fkey2, (400, 400))

        conv_key, key = jr.split(key, 2)
        self.conv_layer = ConvLayer(
            input_dim=input_dim, output_dim=output_dim, key=conv_key
        )

    def _fusion(self, adjacency: Array, control_gradient: Array) -> Array:
        """Fuses the adjacency matrix and control gradient by premultiplication with fusion matrices.

        Args:
            adjacency (Array): Adjacency matrix.
            control_gradient (Array): Control gradient matrix.

        Returns:
            Array: Fused adjacency matrix.
        """
        return self.fusion_1 @ adjacency + self.fusion_2 @ control_gradient

    def __call__(
        self, node_feats: Array, adj_matrix: Array, control_gradient: Array
    ) -> Array:
        """Forward pass of the convolutional equivalent fusion layer.

        Args:
            node_feats (Array): Node features.
            adj_matrix (Array): Adjacency matrix.
            control_gradient (Array): Control gradient matrix.
        Returns:
            Array: Output features after convolution.
        """
        fused_adjacency = self._fusion(adj_matrix, control_gradient)

        out = self.conv_layer(node_feats, fused_adjacency)
        return out
