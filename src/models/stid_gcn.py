import math
import jax
import jax.numpy as jnp
import equinox as eqx


# ======= GLU =======
class GLU(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    dropout_layer: eqx.nn.Dropout
    dropout_rate: float

    def __init__(self, features: int, dropout: float = 0.1, *, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=(1, 1), key=key1
        )
        self.conv2 = eqx.nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=(1, 1), key=key2
        )
        self.conv3 = eqx.nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=(1, 1), key=key3
        )
        # We create a dropout layer. During the __call__, a dropout key must be provided.
        self.dropout_layer = eqx.nn.Dropout(p=dropout)
        self.dropout_rate = dropout

    def __call__(self, x, *, key, train: bool = True):
        # Split key for dropout separately.
        dropout_key, _ = jax.random.split(key)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * jax.nn.sigmoid(x2)
        out = self.dropout_layer(out, key=dropout_key)
        out = self.conv3(out)
        return out


# ======= TemporalEmbedding =======
class TemporalEmbedding(eqx.Module):
    time: int
    time_day: jnp.ndarray  # shape: (time, features)
    time_week: jnp.ndarray  # shape: (7, features)

    def __init__(self, time: int, features: int, *, key):
        key1, key2 = jax.random.split(key)
        self.time = time
        glorot = jax.nn.initializers.glorot_uniform()
        self.time_day = glorot(key1, (time, features))
        self.time_week = glorot(key2, (7, features))

    def __call__(self, x):
        # Expecting x to have at least 3 dimensions. Here we assume that:
        #   - x[..., 1] contains “day” indices (scaled by self.time)
        #   - x[..., 2] contains “week” indices.
        day_emb = x[..., 1]
        # Cast indices to int32 (after scaling by self.time)
        day_indices = jnp.asarray(day_emb * self.time, dtype=jnp.int32)
        time_day = self.time_day[day_indices]
        # Transpose as in PyTorch (assumed to swap channel dimensions)
        time_day = jnp.transpose(time_day, (0, 2, 1))

        week_emb = x[..., 2]
        week_indices = jnp.asarray(week_emb, dtype=jnp.int32)
        time_week = self.time_week[week_indices]
        time_week = jnp.transpose(time_week, (0, 2, 1))

        tem_emb = time_day + time_week
        # Permute dimensions to (batch, channels, height, width)
        tem_emb = jnp.transpose(tem_emb, (0, 3, 1, 2))
        return tem_emb


# ======= Diffusion_GCN =======
class Diffusion_GCN(eqx.Module):
    diffusion_step: int
    conv: eqx.nn.Conv2d
    dropout_layer: eqx.nn.Dropout

    def __init__(
        self, channels: int = 128, diffusion_step: int = 1, dropout: float = 0.1, *, key
    ):
        self.diffusion_step = diffusion_step
        key_conv, key_dropout = jax.random.split(key)
        self.conv = eqx.nn.Conv2d(
            in_channels=diffusion_step * channels,
            out_channels=channels,
            kernel_size=(1, 1),
            key=key_conv,
        )
        self.dropout_layer = eqx.nn.Dropout(p=dropout)

    def __call__(self, x, adj, *, key, train: bool = True):
        out_list = []
        current = x
        # For each diffusion step, apply an einsum based on the dimensionality of adj.
        for i in range(self.diffusion_step):
            if adj.ndim == 3:
                current = jnp.einsum("cnt,nm->cmt", current, adj)
                out_list.append(current)
            elif adj.ndim == 2:
                current = jnp.einsum("cnt,nm->cmt", current, adj)
                out_list.append(current)
        x_cat = jnp.concatenate(out_list, axis=1)
        x_conv = self.conv(x_cat)
        dropout_key, _ = jax.random.split(key)
        output = self.dropout_layer(x_conv, key=dropout_key)
        return output


# ======= Graph_Generator =======
class Graph_Generator(eqx.Module):
    memory: jnp.ndarray  # shape: (channels, num_nodes)
    fc: eqx.nn.Linear

    def __init__(
        self,
        channels: int = 128,
        num_nodes: int = 170,
        diffusion_step: int = 1,
        dropout: float = 0.1,
        *,
        key,
    ):
        key1, key2 = jax.random.split(key)
        glorot = jax.nn.initializers.glorot_uniform()
        self.memory = glorot(key1, (channels, num_nodes))
        self.fc = eqx.nn.Linear(in_features=2, out_features=1, key=key2)

    def __call__(self, x):
        # Calculate the first dynamic adjacency.
        adj_dyn_1 = jnp.einsum("cnt,cm->nm", x, self.memory) / jnp.sqrt(x.shape[1])
        adj_dyn_1 = jax.nn.relu(adj_dyn_1)
        adj_dyn_1 = jax.nn.softmax(adj_dyn_1, axis=-1)

        # Compute the second dynamic adjacency by summing over the last dimension.
        x_sum = x.sum(axis=-1)
        adj_dyn_2 = jnp.einsum("cn,cm->nm", x_sum, x_sum) / jnp.sqrt(x.shape[1])
        adj_dyn_2 = jax.nn.relu(adj_dyn_2)
        adj_dyn_2 = jax.nn.softmax(adj_dyn_2, axis=-1)

        # Concatenate the two adjacencies along a new last axis.
        adj_dyn_1_unsq = jnp.expand_dims(adj_dyn_1, axis=-1)
        adj_dyn_2_unsq = jnp.expand_dims(adj_dyn_2, axis=-1)
        adj_f = jnp.concatenate([adj_dyn_1_unsq, adj_dyn_2_unsq], axis=-1)

        adj_f = jax.vmap(jax.vmap(self.fc))(adj_f)
        # Remove the last singleton dimension.
        adj_f = jnp.squeeze(adj_f, axis=-1)
        adj_f = jax.nn.softmax(adj_f, axis=-1)

        # Apply top-k filtering.
        k = int(adj_f.shape[-1] * 0.8)
        topk_values, topk_indices = jax.lax.top_k(adj_f, k)
        mask = jnp.zeros_like(adj_f)
        # For each batch element, scatter ones at the top-k indices.
        # (Assuming adj_f is 2D [batch, m]. Adjust using vmap if needed.)
        batch_indices = jnp.arange(adj_f.shape[0])[:, None]
        mask = mask.at[batch_indices, topk_indices].set(1)
        adj_f = adj_f * mask
        return adj_f


# ======= DGCN =======
class DGCN(eqx.Module):
    conv: eqx.nn.Conv2d
    generator: Graph_Generator
    gcn: Diffusion_GCN
    emb: jnp.ndarray  # external embedding parameter

    def __init__(
        self,
        channels: int = 128,
        num_nodes: int = 170,
        diffusion_step: int = 1,
        dropout: float = 0.1,
        emb=None,
        *,
        key,
    ):
        key_conv, key_gen, key_gcn = jax.random.split(key, 3)
        self.conv = eqx.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            key=key_conv,
        )
        self.generator = Graph_Generator(
            channels, num_nodes, diffusion_step, dropout, key=key_gen
        )
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout, key=key_gcn)
        self.emb = emb  # supplied externally (or can be None)

    def __call__(self, x, *, key, train: bool = True):
        skip = x
        x_conv = self.conv(x)
        adj_dyn = self.generator(x_conv)
        key_gcn, _ = jax.random.split(key)
        x_gcn = self.gcn(x_conv, adj_dyn, key=key_gcn, train=train)
        return x_gcn * self.emb + skip


# ======= Splitting =======
class Splitting(eqx.Module):
    def __call__(self, x):
        # Assumes x is of shape (batch, channels, height, width)
        return x[..., ::2], x[..., 1::2]


# ======= IDGCN =======
class IDGCN(eqx.Module):
    dropout: float
    num_nodes: int
    splitting: bool
    split: Splitting
    conv1: eqx.nn.Sequential
    conv2: eqx.nn.Sequential
    conv3: eqx.nn.Sequential
    conv4: eqx.nn.Sequential
    dgcn: DGCN

    def __init__(
        self,
        channels: int = 64,
        diffusion_step: int = 1,
        splitting: bool = True,
        num_nodes: int = 170,
        dropout: float = 0.2,
        emb=None,
        *,
        key,
    ):
        # Allocate keys for the four convolution blocks and the DGCN.
        keys = jax.random.split(key, 8)
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.split = Splitting()

        # Define padding parameters.
        pad_l = 3
        pad_r = 3
        k1 = 5
        k2 = 3

        # Define a helper padding function (replication padding on the width dimension).
        def pad_fn(x):
            # Define padding: (dim0, dim1, dim2)
            # Pad the last dimension (time_steps) by 3 to the left and right
            pad_width = ((0, 0), (0, 0), (pad_l, pad_r))

            # Apply replication padding
            x = jnp.pad(x, pad_width=pad_width, mode="edge")

            return x

        def leaky_relu_fn(x):
            return jax.nn.leaky_relu(x, negative_slope=0.01)

        # Each conv block is defined as a Sequential module.
        self.conv1 = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(pad_fn),
                eqx.nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, k1),
                    key=keys[0],
                ),
                eqx.nn.Dropout(p=dropout),
                eqx.nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, k2),
                    key=keys[1],
                ),
                eqx.nn.Lambda(jax.nn.tanh),
            ]
        )
        self.conv2 = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(pad_fn),
                eqx.nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, k1),
                    key=keys[2],
                ),
                eqx.nn.Lambda(leaky_relu_fn),
                eqx.nn.Dropout(p=dropout),
                eqx.nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, k2),
                    key=keys[3],
                ),
                eqx.nn.Lambda(jax.nn.tanh),
            ]
        )
        self.conv3 = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(pad_fn),
                eqx.nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, k1),
                    key=keys[4],
                ),
                eqx.nn.Lambda(leaky_relu_fn),
                eqx.nn.Dropout(p=dropout),
                eqx.nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, k2),
                    key=keys[5],
                ),
                eqx.nn.Lambda(jax.nn.tanh),
            ]
        )
        self.conv4 = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(pad_fn),
                eqx.nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, k1),
                    key=keys[6],
                ),
                eqx.nn.Lambda(leaky_relu_fn),
                eqx.nn.Dropout(p=dropout),
                eqx.nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, k2),
                    key=keys[7],
                ),
                eqx.nn.Lambda(jax.nn.tanh),
            ]
        )
        self.dgcn = DGCN(
            channels=channels,
            num_nodes=num_nodes,
            diffusion_step=diffusion_step,
            dropout=dropout,
            emb=emb,
            key=keys[8],
        )

    def __call__(self, x, *, key, train: bool = True):
        # Split key into several for the sequential blocks and dgcn.
        keys = jax.random.split(key, 9)
        if self.splitting:
            x_even, x_odd = self.split(x)
        else:
            x_even = x
            x_odd = x

        x1 = self.conv1(x_even, key=keys[0])
        x1 = self.dgcn(x1, key=keys[1], train=train)

        d = x_odd * jnp.tanh(x1)

        x2 = self.conv2(x_odd, key=keys[2])
        x2 = self.dgcn(x2, key=keys[3], train=train)
        c = x_even * jnp.tanh(x2)

        x3 = self.conv3(c, key=keys[4])
        x3 = self.dgcn(x3, key=keys[5], train=train)
        x_odd_update = d + x3

        x4 = self.conv4(d, key=keys[6])
        x4 = self.dgcn(x4, key=keys[7], train=train)
        x_even_update = c + x4

        return x_even_update, x_odd_update


# ======= IDGCN_Tree =======
class IDGCN_Tree(eqx.Module):
    memory1: jnp.ndarray  # shape: (channels, num_nodes, 6)
    memory2: jnp.ndarray  # shape: (channels, num_nodes, 3)
    IDGCN1: IDGCN
    IDGCN2: IDGCN
    IDGCN3: IDGCN

    def __init__(
        self,
        channels: int = 64,
        diffusion_step: int = 1,
        num_nodes: int = 170,
        dropout: float = 0.1,
        memory_dim1: int = 6,
        memory_dim2: int = 3,
        *,
        key,
    ):
        # Split key for memory initialization and the IDGCN modules.
        keys = jax.random.split(key, 4)
        glorot = jax.nn.initializers.glorot_uniform()
        self.memory1 = glorot(keys[0], (channels, num_nodes, memory_dim1))
        self.memory2 = glorot(keys[1], (channels, num_nodes, memory_dim2))
        key_idgcn1, key_idgcn2, key_idgcn3 = jax.random.split(keys[3], 3)
        self.IDGCN1 = IDGCN(
            channels=channels,
            diffusion_step=diffusion_step,
            splitting=True,
            num_nodes=num_nodes,
            dropout=dropout,
            emb=self.memory1,
            key=key_idgcn1,
        )
        self.IDGCN2 = IDGCN(
            channels=channels,
            diffusion_step=diffusion_step,
            splitting=True,
            num_nodes=num_nodes,
            dropout=dropout,
            emb=self.memory2,
            key=key_idgcn2,
        )
        self.IDGCN3 = IDGCN(
            channels=channels,
            diffusion_step=diffusion_step,
            splitting=True,
            num_nodes=num_nodes,
            dropout=dropout,
            emb=self.memory2,
            key=key_idgcn3,
        )

    def concat(self, even, odd):
        # Permute dimensions: from (batch, channels, height, width) to (width, channels, height, batch)
        even = jnp.transpose(even, (2, 0, 1))
        odd = jnp.transpose(odd, (2, 0, 1))
        n = even.shape[0]
        cat_list = []
        for i in range(n):
            # Expand dims to simulate unsqueeze.
            cat_list.append(even[i][None, ...])
            cat_list.append(odd[i][None, ...])
        concatenated = jnp.concatenate(cat_list, axis=0)
        concatenated = jnp.transpose(concatenated, (1, 2, 0))

        return concatenated

    def __call__(self, x, *, key, train: bool = True):
        key1, key2, key3 = jax.random.split(key, 3)
        x_even_update1, x_odd_update1 = self.IDGCN1(x, key=key1, train=train)
        x_even_update2, x_odd_update2 = self.IDGCN2(
            x_even_update1, key=key2, train=train
        )
        x_even_update3, x_odd_update3 = self.IDGCN3(
            x_odd_update1, key=key3, train=train
        )
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        output = concat0 + x

        return output


# ======= STIDGCN =======
class STIDGCN(eqx.Module):
    num_nodes: int
    num_time_steps: int
    output_len: int
    Temb: TemporalEmbedding
    start_conv: eqx.nn.Conv2d
    tree: IDGCN_Tree
    glu: GLU
    regression_layer: eqx.nn.Conv2d

    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        num_time_steps: int,
        channels: int,
        output_len: int,
        granularity: int,
        dropout: float = 0.1,
        memory_dim1: int = 128,
        memory_dim2: int = 64,
        *,
        key,
    ):
        keys = jax.random.split(key, 6)
        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps
        self.output_len = output_len
        diffusion_step = 1

        self.Temb = TemporalEmbedding(granularity, channels, key=keys[0])
        self.start_conv = eqx.nn.Conv2d(
            in_channels=input_dim,
            out_channels=channels,
            kernel_size=(1, 1),
            key=keys[1],
        )
        self.tree = IDGCN_Tree(
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout,
            memory_dim1=memory_dim1,
            memory_dim2=memory_dim2,
            key=keys[2],
        )
        self.glu = GLU(channels, dropout=dropout, key=keys[3])
        # self.regression_layer = eqx.nn.Conv2d(in_channels=channels, out_channels=self.output_len, kernel_size=(1, self.output_len), key=keys[4])
        self.regression_layer = eqx.nn.Conv2d(
            in_channels=channels,
            out_channels=self.output_len,
            kernel_size=(1, num_time_steps),
            key=keys[4],
        )

    def param_num(self):
        # Count total number of parameters.
        leaves, _ = eqx.partition(self, eqx.is_array)
        return sum(jnp.size(x) for x in jax.tree_util.tree_leaves(leaves))

    def __call__(self, input, key, train: bool = True):
        # Note: in the original model, the time embedding is computed from a permuted version of the input.

        # input = jnp.expand_dims(input, axis=0)
        input = jnp.transpose(
            input, (2, 1, 0)
        )  # dimension now is (feature_dim=1, num_nodes, time_steps)

        # time_emb = self.Temb(jnp.transpose(input, (0, 3, 2, 1)))
        x_start = self.start_conv(
            input
        )  # dimension now is (feature_dim=32, num_nodes, time_steps)

        # Concatenate along the channel dimension.
        # x = jnp.concatenate([x_start, time_emb], axis=1)
        x = jnp.concatenate([x_start], axis=1)

        key_tree, key_glu, key_reg = jax.random.split(key, 3)
        x_tree = self.tree(x, key=key_tree, train=train)

        gcn_out = self.glu(x_tree, key=key_glu, train=train) + x_tree

        # x_relu = jax.nn.relu(gcn_out)
        prediction = self.regression_layer(gcn_out)
        predictions = jnp.squeeze(prediction, axis=-1)
        return jnp.transpose(predictions, (1, 0))
