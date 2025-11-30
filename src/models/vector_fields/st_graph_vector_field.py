import equinox as eqx
import jax
import jax.numpy as jnp


class STGraphVectorField(eqx.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    num_nodes: int

    cheb_k: int
    node_embed_dim: int

    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear

    # Parameters for AGC
    g_type: str = "agc"
    node_embeddings: jnp.ndarray = None
    weights_pool: jnp.ndarray = None
    bias_pool: jnp.ndarray = None

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        node_embed_dim: int,
        num_nodes: int,
        cheb_k: int,
        *,
        key,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.node_embed_dim = node_embed_dim
        self.num_nodes = num_nodes
        self.cheb_k = cheb_k
        self.g_type = "agc"

        # Split keys for initializing layers
        key, key_linear_in, key_linear_out = jax.random.split(key, 3)
        self.linear_in = eqx.nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
            key=key_linear_in,
        )
        # Final linear maps to hidden_channels * hidden_channels.
        self.linear_out = eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
            key=key_linear_out,
        )

        if self.g_type == "agc":
            # Split keys for AGC parameters
            key, key_node_emb, key_weights, key_bias = jax.random.split(key, 4)
            self.node_embeddings = jax.random.normal(
                key_node_emb, (num_nodes, node_embed_dim)
            )
            self.weights_pool = jax.random.normal(
                key_weights, (node_embed_dim, cheb_k, hidden_dim, hidden_dim)
            )
            self.bias_pool = jax.random.normal(key_bias, (node_embed_dim, hidden_dim))
        else:
            raise ValueError("Check g_type argument")

    def __call__(self, z: jnp.ndarray):
        """
        Args:
          z: a JAX array with shape (..., hidden_channels)
             Typically, z will have shape (batch, num_nodes, hidden_channels)
        """
        # First linear transformation and activation
        z = jax.vmap(self.linear_in)(z)
        z = jax.vmap(jax.nn.relu)(z)

        if self.g_type == "agc":
            z = self.agc(z)
        else:
            raise ValueError("Check g_type argument")

        # Final linear transformation and reshape.
        z = jax.vmap(self.linear_out)(z)
        z = jax.vmap(jnp.tanh)(z)
        return z

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Computes node adaptive parameter learning and data adaptive graph generation.
        Args:
          z: JAX array with shape (batch, num_nodes, hidden_hidden_channels)
        Returns:
          Updated z after applying adaptive graph convolution.
        """
        node_num = self.node_embeddings.shape[0]
        # Compute similarity and generate supports:
        sim = jnp.dot(self.node_embeddings, self.node_embeddings.T)
        supports = jax.nn.softmax(jax.nn.relu(sim), axis=1)

        # Use laplacian=False: support_set = [I, supports]
        eye = jnp.eye(node_num)
        support_set = [eye, supports]

        # Compute Chebyshev polynomials for k = 2, ..., cheb_k-1
        for k in range(2, self.cheb_k):
            next_support = jnp.matmul(2 * supports, support_set[-1]) - support_set[-2]
            support_set.append(next_support)
        # Stack supports to shape (cheb_k, num_nodes, num_nodes)
        supports = jnp.stack(support_set, axis=0)

        # Compute weights and bias based on node embeddings:
        weights = jnp.einsum("nd,dkio->nkio", self.node_embeddings, self.weights_pool)
        bias = jnp.dot(self.node_embeddings, self.bias_pool)

        # Graph convolution:
        # x_g: shape (cheb_k, num_nodes, hidden_hidden_channels)
        x_g = jnp.einsum("knm,mc->knc", supports, z)
        # Permute to (num_nodes, cheb_k, hidden_hidden_channels)
        x_g = jnp.transpose(x_g, (1, 0, 2))
        # Final weighted aggregation and add bias:
        # z: shape (num_nodes, hidden_hidden_channels)
        z = jnp.einsum("nki,nkio->no", x_g, weights) + bias
        return z
