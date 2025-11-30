# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830

import math
import torch
from torch.utils.data import Dataset
import jax
import diffrax
import jax.numpy as jnp
import numpy as np
import jax.random as jr


def split_data(data, window_size):
    length = len(data)
    split_data = []
    for i in range(0, length - window_size):
        split_data.append(data[i : i + window_size + 1])
    return split_data


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


class LPDataset(Dataset):
    def __init__(self, path, window_size):
        super(LPDataset, self).__init__()
        self.data = torch.from_numpy(np.load(path))
        self.window_size = window_size
        self.num = self.data.size(0) - window_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return (
            self.data[item : item + self.window_size],
            self.data[item + self.window_size],
        )


class SDataset(Dataset):
    def __init__(
        self,
        mode,
        path="/mnt/sdd/MSc_projects/torben/projects/graph-neural-cdes-main/processed",
    ):
        train_samples = 5000
        test_offset = 75000
        test_samples = 5000
        if mode == "train":
            self.adjs = np.load(path + "/train_mats.npy").astype(np.float32)[
                :train_samples
            ]
            self.atom_forces = np.load(path + "/train_atom_forces.npy").astype(
                np.float32
            )[:train_samples]
            self.dfts = np.load(path + "/train_dfts.npy").astype(np.float32)[
                :train_samples
            ]
            self.atom_lists = np.load(path + "/train_atom_lists.npy").astype(
                np.float32
            )[:train_samples]
        if mode == "val":
            self.adjs = np.load(path + "/train_mats.npy").astype(np.float32)[
                test_offset : test_offset + test_samples
            ]
            self.atom_forces = np.load(path + "/train_atom_forces.npy").astype(
                np.float32
            )[test_offset : test_offset + test_samples]
            self.dfts = np.load(path + "/train_dfts.npy").astype(np.float32)[
                test_offset : test_offset + test_samples
            ]
            self.atom_lists = np.load(path + "/train_atom_lists.npy").astype(
                np.float32
            )[test_offset : test_offset + test_samples]

        self.atom_lists = torch.from_numpy(self.atom_lists[:, 0, :])

        self.num = self.adjs.shape[0]
        self.dimensions = torch.from_numpy(self.adjs).size()

        length = 10
        self.ts = torch.linspace(0.0, 4.5, length)

        dataset_size = self.adjs.shape[0]
        window_length = self.adjs.shape[1]
        num_nodes = self.adjs.shape[3]
        ts = jnp.broadcast_to(
            jnp.linspace(0, 5.0, window_length), (dataset_size, window_length)
        )  # (dataset_size, window_length)
        ts_bc = jnp.repeat(
            jnp.repeat(ts[:, :, None, None], num_nodes, axis=2), num_nodes, axis=3
        )  # broadcast to shape (dataset_size, window_length, num_nodes, num_nodes)
        ys = jnp.stack(
            [ts_bc, self.adjs], axis=-1
        )  # time is a channel (dataset_size, window_length, num_nodes, num_nodes, 2)
        self.coeffs = torch.from_numpy(
            np.asarray(jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys))
        ).permute(1, 0, 2, 3, 4, 5)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return (
            self.ts,
            self.coeffs[item],
            self.atom_forces[item],
            self.dfts[item],
            self.atom_lists[item],
        )

    def data_dimensions(self):
        return list(self.dimensions)


class MD17Dataset(Dataset):
    def __init__(
        self,
        mode,
        path="/mnt/sdd/MSc_projects/torben/dataset/md17_aspirin",
    ):
        train_samples = 1000
        val_offset = 35000
        val_samples = 1000
        test_offset = 75000
        test_samples = 5000
        if mode == "train":
            self.adjs = np.load(path + "/adjacencies.npy").astype(np.float32)[
                :train_samples
            ]
            self.atom_forces = np.load(path + "/atom_forces.npy").astype(np.float32)[
                :train_samples
            ]
            self.positions = np.load(path + "/positions.npy").astype(np.float32)[
                :train_samples
            ]
            self.dfts = np.load(path + "/dfts.npy").astype(np.float32)[:train_samples]
        if mode == "val":
            self.adjs = np.load(path + "/adjacencies.npy").astype(np.float32)[
                val_offset : val_offset + val_samples
            ]
            self.atom_forces = np.load(path + "/atom_forces.npy").astype(np.float32)[
                val_offset : val_offset + val_samples
            ]
            self.positions = np.load(path + "/positions.npy").astype(np.float32)[
                val_offset : val_offset + val_samples
            ]
            self.dfts = np.load(path + "/dfts.npy").astype(np.float32)[
                val_offset : val_offset + val_samples
            ]
        if mode == "test":
            self.adjs = np.load(path + "/adjacencies.npy").astype(np.float32)[
                test_offset : test_offset + test_samples
            ]
            self.atom_forces = np.load(path + "/atom_forces.npy").astype(np.float32)[
                test_offset : test_offset + test_samples
            ]
            self.positions = np.load(path + "/positions.npy").astype(np.float32)[
                test_offset : test_offset + test_samples
            ]
            self.dfts = np.load(path + "/dfts.npy").astype(np.float32)[
                test_offset : test_offset + test_samples
            ]

        self.atoms = np.load(path + "/atoms.npy").astype(np.float32)
        self.num = self.adjs.shape[0]
        self.dimensions = torch.from_numpy(self.adjs).size()

        length = 10
        self.ts = torch.linspace(0.0, 5.0, length)

        dataset_size = self.adjs.shape[0]
        window_length = self.adjs.shape[1]
        num_nodes = self.adjs.shape[3]
        ts = jnp.broadcast_to(
            jnp.linspace(0, 5.0, window_length), (dataset_size, window_length)
        )  # (dataset_size, window_length)
        ts_bc_adj = jnp.repeat(
            jnp.repeat(ts[:, :, None, None], num_nodes, axis=2), num_nodes, axis=3
        )  # broadcast to shape (dataset_size, window_length, num_nodes, num_nodes)
        # ts_bc_pos = jnp.repeat(
        #     jnp.repeat(ts[:, :, None, None], num_nodes, axis=2), 3, axis=3
        # )  # broadcast to shape (dataset_size, window_length, num_nodes, num_nodes)
        ys_adj = jnp.stack(
            [ts_bc_adj, self.adjs], axis=-1
        )  # time is a channel (dataset_size, window_length, num_nodes, num_nodes, 2)
        # ys_pos = jnp.stack(
        #     [ts_bc_pos, self.positions], axis=-1
        # )  # time is a channel (dataset_size, window_length, num_nodes, num_nodes, 2)
        self.coeffs_adj = torch.from_numpy(
            np.asarray(jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys_adj))
        ).permute(1, 0, 2, 3, 4, 5)
        # self.coeffs_pos = torch.from_numpy(
        #     np.asarray(jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys_pos))
        # ).permute(1, 0, 2, 3, 4, 5)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return (
            self.ts,
            self.coeffs_adj[item],
            self.atom_forces[item],
            self.dfts[item],
            self.atoms,
        )

    def data_dimensions(self):
        return list(self.dimensions)


def get_iso17_dataset(
    mode="train",
    path="/mnt/sdd/MSc_projects/torben/projects/graph-neural-cdes-main/processed",
):
    length = 10
    if mode == "train":
        adjs = np.load(path + "/train_mats.npy").astype(np.float32)[:7500]
        atom_forces = np.load(path + "/train_atom_forces.npy").astype(np.float32)[:7500]
        dfts = np.load(path + "/train_dfts.npy").astype(np.float32)[:7500]
    if mode == "val":
        adjs = np.load(path + "/train_mats.npy").astype(np.float32)[750:1500]
        atom_forces = np.load(path + "/train_atom_forces.npy").astype(np.float32)[
            750:1500
        ]
        dfts = np.load(path + "/train_dfts.npy").astype(np.float32)[750:1500]
    dataset_size = adjs.shape[0]
    window_length = adjs.shape[1]
    num_nodes = adjs.shape[3]

    ts = jnp.broadcast_to(
        jnp.linspace(0, 5.0, window_length), (dataset_size, window_length)
    )  # (dataset_size, window_length)
    ts_bc = jnp.repeat(
        jnp.repeat(ts[:, :, None, None], num_nodes, axis=2), num_nodes, axis=3
    )  # broadcast to shape (dataset_size, window_length, num_nodes, num_nodes)
    ys = jnp.stack(
        [ts_bc, adjs], axis=-1
    )  # time is a channel (dataset_size, window_length, num_nodes, num_nodes, 2)
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)

    return ts, coeffs, atom_forces


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    perm = jr.permutation(key, indices)
    (key,) = jr.split(key, 1)
    start = 0
    end = batch_size
    while end < dataset_size:
        batch_perm = perm[start:end]
        yield tuple(array[batch_perm] for array in arrays)
        start = end
        end = start + batch_size


def MissRate(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask1 = (input > 0) & (target == 0)
    mask2 = (input == 0) & (target > 0)
    mask = mask1 | mask2
    return mask.sum().item() / num


def graph_mini_batch(adj_matrix_list, x_list, batch_size=64):
    # Implement the function here
    num_full_batches = math.floor(len(adj_matrix_list) / batch_size)
    for batch in range(num_full_batches):

        adjs = torch.clone(
            adj_matrix_list[batch * batch_size : (batch + 1) * batch_size]
        )
        A_B = []
        for i in range(10):
            append = torch.block_diag(*(adjs[:, i, :, :])).unsqueeze(0)
            A_B.append(append)
        A_B = torch.cat(A_B)

        X_B = torch.vstack(
            [
                s.permute(1, 0, 2)
                for s in [*(x_list[batch * batch_size : (batch + 1) * batch_size])]
            ]
        ).permute(1, 0, 2)
        Batch = torch.vstack([torch.full((19, 1), i) for i in range(batch_size)])
        yield A_B, X_B, Batch

    if (len(adj_matrix_list) % batch_size) != 0:
        adjs = torch.clone(adj_matrix_list[num_full_batches * batch_size :])
        A_B = []
        for i in range(10):
            append = torch.block_diag(*(adjs[:, i, :, :])).unsqueeze(0)
            A_B.append(append)
        A_B = torch.cat(A_B)
        X_B = torch.vstack(
            [s.permute(1, 0, 2) for s in [*(x_list[num_full_batches * batch_size :])]]
        ).permute(1, 0, 2)

        Batch = torch.vstack(
            [torch.full((19, 1), i) for i in range(len(adj_matrix_list) % batch_size)]
        )
        yield A_B, X_B, Batch
