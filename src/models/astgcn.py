import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Sequence


class SpatialAttention(eqx.Module):
    """
    Compute spatial attention scores
    """

    W_1: jnp.ndarray  # shape (T,)
    W_2: jnp.ndarray  # shape (F, T)
    W_3: jnp.ndarray  # shape (F,)
    b_s: jnp.ndarray  # shape (1, N, N)
    V_s: jnp.ndarray  # shape (N, N)

    def __init__(
        self,
        num_vertices: int,
        num_features: int,
        num_timesteps: int,
        *,
        key: jax.random.PRNGKey,
    ):
        # split key for each parameter
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.W_1 = jax.random.normal(k1, (num_timesteps,))
        self.W_2 = jax.random.normal(k2, (num_features, num_timesteps))
        self.W_3 = jax.random.normal(k3, (num_features,))
        self.b_s = jax.random.normal(k4, (num_vertices, num_vertices))
        self.V_s = jax.random.normal(k5, (num_vertices, num_vertices))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, N, F, T)
        N, F, T = x.shape
        # lhs: (batch, N, T)
        tmp = jnp.tensordot(x, self.W_1, axes=([2], [0]))  # -> (N, F)
        lhs = jnp.tensordot(tmp, self.W_2, axes=([1], [0]))  # -> (N, T)
        # rhs: (batch, T, N)
        x_trans = x.transpose(1, 2, 0)  # (F, T, N)
        rhs = jnp.tensordot(self.W_3, x_trans, axes=([0], [0]))  # -> (T, N)
        # product: (batch, N, N)
        product = jnp.matmul(lhs, rhs)
        # attention with parameters
        S_inter = jax.nn.sigmoid(product + self.b_s)  # (N, N)
        S = jnp.einsum("ij,jk->ik", self.V_s, S_inter)
        # normalize
        S = S - jnp.max(S, axis=1, keepdims=True)
        exp_S = jnp.exp(S)
        S_normalized = exp_S / jnp.sum(exp_S, axis=1, keepdims=True)
        return S_normalized


class ChebConvWithSAT(eqx.Module):
    """
    K-order Chebyshev graph convolution with spatial attention
    """

    Theta: jnp.ndarray  # shape (K, F, num_filters)
    num_filters: int
    K: int

    def __init__(
        self,
        num_features: int,
        num_filters: int,
        K: int,
        *,
        key: jax.random.PRNGKey,
    ):
        k1 = key
        # initialize Theta
        self.Theta = jax.random.normal(k1, (K, num_features, num_filters))
        self.num_filters = num_filters
        self.K = K

        # x: (N, F, T), spatial_attention: (N, N)

    def __call__(
        self,
        x: jnp.ndarray,
        spatial_attention: jnp.ndarray,
        node_embeddings: jnp.ndarray,
    ) -> jnp.ndarray:
        node_num = node_embeddings.shape[0]
        # Compute similarity and generate supports:
        sim = jnp.dot(node_embeddings, node_embeddings.T)
        supports = jax.nn.softmax(jax.nn.relu(sim), axis=1)

        N, F, T = x.shape
        outputs = []
        for t in range(T):
            graph_signal = x[..., t]  # (N, F)
            out_t = jnp.zeros((N, self.num_filters))
            cheb_polynomials = []
            for k in range(self.K):
                if k == 0:
                    T_k = jnp.identity(N)
                elif k == 1:
                    T_k = supports
                else:
                    T_k = (
                        2 * supports * cheb_polynomials[k - 1] - cheb_polynomials[k - 2]
                    )
                cheb_polynomials.append(T_k)

                # apply attention
                T_k_at = spatial_attention * T_k  # broadcast (N, N)
                # message passing
                rhs = jnp.matmul(T_k_at.transpose(1, 0), graph_signal)  # (batch, N, F)
                theta_k = self.Theta[k]  # (F, num_filters)
                out_t = out_t + jnp.einsum("if,fo->io", rhs, theta_k)
            outputs.append(out_t[..., None])  # (N, num_filters, 1)
        h = jnp.concatenate(outputs, axis=-1)  # (N, num_filters, T)
        return jax.nn.relu(h)


class TemporalAttention(eqx.Module):
    """
    Compute temporal attention scores
    """

    U_1: jnp.ndarray  # (N,)
    U_2: jnp.ndarray  # (F, N)
    U_3: jnp.ndarray  # (F,)
    b_e: jnp.ndarray  # (1, T, T)
    V_e: jnp.ndarray  # (T, T)

    def __init__(
        self,
        num_vertices: int,
        num_features: int,
        num_timesteps: int,
        *,
        key: jax.random.PRNGKey,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.U_1 = jax.random.normal(k1, (num_vertices,))
        self.U_2 = jax.random.normal(k2, (num_features, num_vertices))
        self.U_3 = jax.random.normal(k3, (num_features,))
        self.b_e = jax.random.normal(k4, (num_timesteps, num_timesteps))
        self.V_e = jax.random.normal(k5, (num_timesteps, num_timesteps))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (N, F, T)
        N, F, T = x.shape
        # lhs: (T, N)
        x_t = x.transpose(2, 1, 0)  # (batch, T, F, N)
        tmp = jnp.tensordot(x_t, self.U_1, axes=([2], [0]))  # -> (T, F)
        lhs = jnp.tensordot(tmp, self.U_2, axes=([1], [0]))  # -> (T, N)
        # rhs: (N, T)
        x_e = x.transpose(1, 0, 2)  # (F, N, T)
        rhs = jnp.tensordot(self.U_3, x_e, axes=([0], [0]))  # -> (N, T)
        # product: (T, T)
        product = jnp.matmul(lhs, rhs)
        E_inter = jax.nn.sigmoid(product + self.b_e)
        E = jnp.einsum("ij,jk->ik", self.V_e, E_inter)
        # normalize
        E = E - jnp.max(E, axis=1, keepdims=True)
        exp_E = jnp.exp(E)
        E_normalized = exp_E / jnp.sum(exp_E, axis=1, keepdims=True)
        return E_normalized


class ASTGCNBlock(eqx.Module):
    """
    Single block of ASTGCN, combining spatial & temporal attention and graph convolution
    """

    SAt: SpatialAttention
    cheb_conv: ChebConvWithSAT
    TAt: TemporalAttention
    time_conv: eqx.nn.Conv2d
    residual_conv: eqx.nn.Conv2d
    ln: eqx.nn.LayerNorm

    def __init__(
        self,
        num_vertices: int,
        num_features: int,
        num_timesteps: int,
        K: int,
        num_chev_filters: int,
        num_time_filters: int,
        time_conv_strides: int,
        *,
        key: jax.random.PRNGKey,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        # modules
        self.SAt = SpatialAttention(
            num_vertices, num_features, time_conv_strides, key=k1
        )  # None placeholder for time in init
        self.TAt = TemporalAttention(
            num_vertices, num_features, time_conv_strides, key=k2
        )
        self.cheb_conv = ChebConvWithSAT(num_features, num_chev_filters, K, key=k3)
        # temporal conv layers: input channels = num_chev_filters, output = num_time_filters
        self.time_conv = eqx.nn.Conv2d(
            in_channels=num_chev_filters,
            out_channels=num_time_filters,
            kernel_size=(1, 3),
            stride=(1, time_conv_strides),
            padding=(0, 1),
            key=k4,
        )
        # residual conv: in_channels = num_features -> out = num_time_filters
        self.residual_conv = eqx.nn.Conv2d(
            in_channels=num_features,
            out_channels=num_time_filters,
            kernel_size=(1, 1),
            stride=(1, time_conv_strides),
            key=k5,
        )
        self.ln = eqx.nn.LayerNorm(
            shape=(num_time_filters,),
        )

    def __call__(self, x: jnp.ndarray, node_embeddings: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, N, F, T)
        N, F, T = x.shape
        # temporal attention
        E = self.TAt(x)
        # apply temporal attention
        x_flat = x.reshape(-1, T)
        x_tatt = jnp.matmul(x_flat, E).reshape(N, F, T)
        # spatial gcn
        S = self.SAt(x_tatt)
        gcn_out = self.cheb_conv(
            x, S, node_embeddings
        )  # (batch, N, num_chev_filters, T)
        # time conv
        gcn_trans = gcn_out.transpose(1, 0, 2)
        t_out = self.time_conv(gcn_trans).transpose(1, 0, 2)
        # residual
        x_res = self.residual_conv(x.transpose(1, 0, 2)).transpose(1, 0, 2)
        # combine
        h = jax.nn.relu(x_res + t_out)
        # layer norm
        h = jnp.squeeze(h)
        return jnp.expand_dims(jax.vmap(self.ln)(h), axis=-1)


class ASTGCNSubmodule(eqx.Module):
    """
    One submodule of ASTGCN
    """

    blocks: Sequence[ASTGCNBlock]
    final_conv: eqx.nn.Conv2d
    W: jnp.ndarray

    def __init__(
        self,
        num_for_prediction: int,
        backbones: List[dict],
        *,
        num_vertices: int,
        num_features: int,
        num_timesteps: int,
        key: jax.random.PRNGKey,
    ):
        # split keys
        keys = jax.random.split(key, len(backbones) + 2)
        block_keys = keys[: len(backbones)]
        fc_key = keys[-2]
        w_key = keys[-1]
        # create blocks
        self.blocks = [
            ASTGCNBlock(
                num_vertices,
                num_features,
                num_timesteps,
                b["K"],
                b["num_of_chev_filters"],
                b["num_of_time_filters"],
                b["time_conv_strides"],
                key=block_keys[i],
            )
            for i, b in enumerate(backbones)
        ]
        last_nf = backbones[-1]["time_conv_strides"]
        # final conv
        self.final_conv = eqx.nn.Conv2d(
            in_channels=last_nf,
            out_channels=num_for_prediction,
            kernel_size=(1, last_nf),
            key=fc_key,
        )
        # W weight for output
        self.W = jax.random.normal(w_key, (num_vertices, num_for_prediction))

    def __call__(self, x: jnp.ndarray, node_embeddings: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, N, F, T)
        # x: (N, F, T)
        h = x
        for block in self.blocks:
            h = block(h, node_embeddings)
        # conv across time
        h_t = self.final_conv(h.transpose(2, 0, 1))  # (pred, N, 1)
        out = h_t[:, :, -1].transpose(1, 0)  # (N, pred)
        return out * self.W


class ASTGCN(eqx.Module):
    """
    ASTGCN model with multiple submodules (hour/day/week)
    """

    submodule: ASTGCNSubmodule
    node_embeddings: jnp.ndarray

    def __init__(
        self,
        num_for_prediction: int,
        backbones: List[dict],
        *,
        num_vertices: int,
        node_embed_dim: int,
        num_features: int,
        num_timesteps: int,
        key: jax.random.PRNGKey,
    ):
        key_sub, key_emb = jax.random.split(key, 2)

        self.submodule = ASTGCNSubmodule(
            num_for_prediction,
            backbones,
            num_vertices=num_vertices,
            num_features=num_features,
            num_timesteps=num_timesteps,
            key=key_sub,
        )

        self.node_embeddings = jax.random.normal(
            key_emb, (num_vertices, node_embed_dim)
        )

    def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        # x_list length must match submodules
        x = x.transpose(
            1, 2, 0
        )  # permute into shape (num_nodes, num_features, num_timesteps)
        out = self.submodule(x, self.node_embeddings)
        return out
