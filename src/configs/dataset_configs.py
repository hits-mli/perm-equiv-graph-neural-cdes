import pydantic
import typing as tp
import json
import hashlib
import pickle
import logging
from pathlib import Path
import math

import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data, TemporalData
from torch_geometric.utils import to_dense_adj
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
import jax
import jax.numpy as jnp
import diffrax

from dataset import ODEDataset
from dataset import misc
from dataset.tgb_dataset import SlidingWindowTemporalLoader

logger = logging.getLogger(__name__)

MINUTE_DURATION = 60
HOUR_DURATION = 60 * MINUTE_DURATION
DAY_DURATION = 24 * HOUR_DURATION
WEEK_DURATION = 7 * DAY_DURATION
MONTH_DURATION = 30 * DAY_DURATION
YEAR_DURATION = 365 * DAY_DURATION


class ODEDataSetCfg(pydantic.BaseModel):
    """
    Configuration class for ODEDataset.
    """

    name: tp.Literal["heat", "gene", "mutualistic", "sir"] = pydantic.Field(
        ...,
        description="Dynamical System we are simulating",
    )
    batch_size: int = pydantic.Field(
        default=1,
        description="Number of graphs to sample",
    )
    dynamic_graph: bool = pydantic.Field(
        ...,
        description="Static or dynamic graph",
    )
    all_dynamic: bool = pydantic.Field(
        ...,
        description="Structural change for training phase only or for the whole sequence",
    )
    graph_type: tp.Literal[
        "grid", "random", "power_law", "small_world", "community", "mixed"
    ] = pydantic.Field(
        default="grid",
        description="Distribution of generated graphs",
    )
    split_ratio: tp.List = pydantic.Field(
        default=[0.8, 0.2], description="train and test ratio"
    )
    num_nodes: int = 400
    final_time: float = 5.0  # terminal time
    time_tick: int = 100  # number of observations
    sampling_type: tp.Literal["irregular", "equal"] = "irregular"

    method: tp.Literal[
        "Dopri5",
        "Tsit5",
    ] = pydantic.Field(default="Dopri5")
    dt0: float = 0.01
    layout: tp.Literal["community", "degree"] = "community"
    operator_type: tp.Literal["lap", "norm_lap", "kipf", "norm_adj"] = "norm_lap"
    padding_mode: tp.Literal["same", "none"] = "same"

    cache_dir: str = pydantic.Field(
        default="cache",
        description="Directory to store cached results",
    )

    interpolation: tp.Literal["linear", "cubic"] = pydantic.Field(
        default="cubic", description="Interpolation method"
    )

    # TODO: make same as general seed
    seed: int = 1234

    amp_range: tp.Tuple = pydantic.Field(
        default=(0.5, 1.0),
        description="Range from which the peak amplitues are sampled at initialisation",
    )

    model_config = pydantic.ConfigDict(extra="forbid")

    beta: tp.Tuple[float, float] = pydantic.Field(
        default=(0.5, 1.0),
        description="Beta parameter for case of dies out and spreading resp.",
    )
    gamma: tp.Tuple[float, float] = pydantic.Field(
        default=(0.3, 1.0),
        description="Gamma parameter for case of dies out and spreading resp.",
    )

    def padding_graph_by_time(
        self,
        adjacencies: jnp.ndarray,
        events_indices: jnp.ndarray,
        ts: jnp.ndarray,
        padding_mode: str = "none",
        padding_last: bool = False,
    ) -> jnp.ndarray:
        """
        Pads the graph adjacency matrices by time, that is we duplicate the adjacency matrix at the time of the event until the next event.

        Args:
            adjacencies (jnp.ndarray): The adjacency matrices.
            events_indices (jnp.ndarray): The indices at which structural changes occur.
            ts (jnp.ndarray): The time stamps.
            padding_mode (str): The padding mode.
            padding_last (bool): Whether to pad the last event.

        Returns:
            jnp.ndarray: The padded adjacency matrices.
        """
        if events_indices is None:
            return jnp.repeat(adjacencies[None, ...], len(ts), axis=0)

        event_idx_arr = jnp.zeros(len(ts), dtype=jnp.int32)
        event_idx_arr = event_idx_arr.at[events_indices].set(1)
        cum_event_idx_arr = jnp.cumsum(event_idx_arr)

        if padding_mode == "none":
            padded_A = jnp.where(
                event_idx_arr > 0, adjacencies[cum_event_idx_arr], jnp.nan
            )
        else:
            padded_A = adjacencies[cum_event_idx_arr]

        if padding_last and events_indices[-1] != len(ts) - 1:
            padded_A = padded_A.at[-1].set(adjacencies[event_idx_arr[-1]])

        return padded_A

    def get_graph_interpolation_coeffs(
        self, ts: jnp.ndarray, padded_adjacencies: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Stacks padded adjacency matrices with time stamps and gets the graph interpolation coefficients.

        Args:
            ts (jnp.ndarray): The time stamps.
            padded_adjacencies (jnp.ndarray): The padded adjacency matrices.

        Returns:
            jnp.ndarray: The interpolation coefficients.
        """
        t_index = jnp.broadcast_to(
            ts[:, None, None],
            (ts.shape[0], padded_adjacencies.shape[1], padded_adjacencies.shape[2]),
        )

        X = jnp.stack([t_index, padded_adjacencies], axis=-1)

        if self.interpolation == "linear":
            coeffs = diffrax.linear_interpolation(ts, X)
        elif self.interpolation == "cubic":
            coeffs = diffrax.backward_hermite_coefficients(ts, X)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")
        return coeffs

    def prepare_graph_path(
        self, ts: jnp.ndarray, adjacencies: jnp.ndarray, events_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Removes events that are not visible in current configuration, then pads adjacencies and calculates interpolation coefficients.

        Args:
            ts (jnp.ndarray): The time points.
            adjacencies (jnp.ndarray): The adjacency matrices.
            events_indices (jnp.ndarray): The indices of events.

        Returns:
            jnp.ndarray: The prepared graph path.
        """
        visible_event_num = jnp.sum(jnp.less(events_indices, ts.shape[1]))
        # truncate A_list and A_t
        adjacencies = adjacencies[:, : visible_event_num + 1, ...]
        events_indices = events_indices[:visible_event_num]

        padded_adjacencies = jax.vmap(
            self.padding_graph_by_time, in_axes=(0, None, 0, None)
        )(adjacencies, events_indices, ts, self.padding_mode)

        coeffs = jax.vmap(self.get_graph_interpolation_coeffs)(ts, padded_adjacencies)
        return coeffs

    def get_interpolation_coeffs(
        self, ts: jnp.ndarray, signal: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Stacks signal matrices with time stamps and gets the graph interpolation coefficients.

        Args:
            ts (jnp.ndarray): The time stamps.
            signal (jnp.ndarray): The signal matrices.

        Returns:
            jnp.ndarray: The interpolation coefficients.
        """
        if self.name == "sir":
            t_index = jnp.broadcast_to(
                ts[:, :, None, None],
                (ts.shape[0], signal.shape[1], signal.shape[2], 3),
            )
        else:
            t_index = jnp.broadcast_to(
                ts[:, :, None],
                (ts.shape[0], signal.shape[1], signal.shape[2]),
            )

        X = jnp.stack([t_index, signal], axis=-1)

        if self.interpolation == "linear":
            coeffs = jax.vmap(diffrax.linear_interpolation)(ts, X)
        elif self.interpolation == "cubic":
            coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, X)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

        return coeffs

    def get_split_train_data(self, dataset: ODEDataset) -> tp.Dict[str, jnp.ndarray]:
        """
        Populates data dictionary with split training data.

        Args:
            dataset (ODEDataset): The dataset.

        Returns:
            tp.Dict[str, jnp.ndarray]: The split training data.
        """
        id_train, id_test_extra, id_test_inter = dataset.split_train_val_test()

        data_dict = {
            "t": jnp.array(dataset.t),
            "train_t": jnp.array(dataset.t[:, id_train]),
            "true_y": jnp.array(dataset.true_y),
            "train_true_y": jnp.array(dataset.true_y[:, id_train]),
            "true_y0": jnp.array(dataset.x0),
            "id_train": jnp.array(id_train),
            "id_test_extra": jnp.array(id_test_extra),
            "id_test_inter": jnp.array(id_test_inter),
            "A": jnp.array(dataset.A),
            "A_t": jnp.array(dataset.events_time_indices),
            "events_time": jnp.array(dataset.events_time),
            "labels": jnp.array(dataset.labels),
        }

        if self.dynamic_graph:
            OM = jax.vmap(
                jax.vmap(misc.get_graph_operator, in_axes=(None, 0, 0)),
                in_axes=(None, 0, 0),
            )(self.operator_type, dataset.A, dataset.L)
        else:
            OM = misc.get_graph_operator(self.operator_type, dataset.A, dataset.L)

        data_dict.update({"A": OM})

        data_dict["train_graph_path_coeffs"] = self.prepare_graph_path(
            dataset.t[:, id_train], data_dict["A"], dataset.events_time_indices
        )
        data_dict["val_graph_path_coeffs"] = self.prepare_graph_path(
            dataset.t, data_dict["A"], dataset.events_time_indices
        )

        # get data interpolation coeficients
        data_dict["y_coeffs"] = self.get_interpolation_coeffs(dataset.t, dataset.true_y)

        return data_dict

    def get_split_test_data(self, dataset: ODEDataset) -> tp.Dict[str, jnp.ndarray]:
        """
        Populates data dictionary with split test data.

        Args:
            dataset (ODEDataset): The dataset.

        Returns:
            tp.Dict[str, jnp.ndarray]: The split test data.
        """
        data_dict = {
            "t": jnp.array(dataset.t),
            "true_y": jnp.array(dataset.true_y),
            "true_y0": jnp.array(dataset.x0),
            "A": jnp.array(dataset.A),
            "A_t": jnp.array(dataset.events_time_indices),
            "events_time": jnp.array(dataset.events_time),
            "labels": jnp.array(dataset.labels),
        }

        if self.dynamic_graph:
            OM = jnp.vectorize(
                lambda A, L: misc.get_graph_operator(self.operator_type, A, L),
                signature="(n,n),(n,n)->(n,n)",
            )(jnp.array(dataset.A), jnp.array(dataset.L))
        else:
            OM = misc.get_graph_operator(self.operator_type, dataset.A, dataset.L)

        data_dict.update({"A": OM})

        data_dict["test_graph_path_coeffs"] = self.prepare_graph_path(
            dataset.t, data_dict["A"], dataset.events_time_indices
        )

        # get data interpolation coeficients
        data_dict["y_coeffs"] = self.get_interpolation_coeffs(dataset.t, dataset.true_y)

        return data_dict

    def _hash_config(self) -> str:
        """
        Hashes the configuration.

        Returns:
            str: The hashed configuration.
        """
        config_dict = self.model_dump(exclude={"cache_dir"})
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self, config_hash: str, data_key: str) -> Path:
        """
        Gets the cache path for the given configuration hash and data key.

        Args:
            config_hash (str): The configuration hash.
            data_key (str): The data key.

        Returns:
            Path: The cache path.
        """
        # Create cache directory if it doesn't exist
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a filename that includes both the config hash and data_key
        cache_filename = f"cache_{config_hash}_{data_key}.pkl"
        return cache_dir / cache_filename

    def get_training_data(self, train_data_key: str) -> tp.Dict[str, jnp.ndarray]:
        """
        Checks if train data exists in the cache. If not, computes it and saves it to the cache.

        Args:
            train_data_key (str): The train data key.

        Returns:
            tp.Dict[str, jnp.ndarray]: The train data.
        """
        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, f"{train_data_key}_train")

        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded training cache from {cache_path}")
                    return pickle.load(f)
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        # Compute if not cached
        train_dataset = ODEDataset(self)
        train_data_dict = self.get_split_train_data(train_dataset)

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                logger.info(f"Saved training cache to {cache_path}")
                pickle.dump(train_data_dict, f)
        except Exception as e:
            logger.info(f"Cache save failed: {e}")

        return train_data_dict

    def get_validation_data(self, val_data_key: str) -> tp.Dict[str, jnp.ndarray]:
        """
        Checks if train data exists in the cache. If not, computes it and saves it to the cache.

        Args:
            val_data_key (str): The validation data key.

        Returns:
            tp.Dict[str, jnp.ndarray]: The validation data.
        """
        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, f"{val_data_key}_test")

        # TODO: fix crude workaround
        self.seed += 500
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded validation cache from {cache_path}")
                    return pickle.load(f)
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        val_dataset = ODEDataset(self)
        val_data_dict = self.get_split_test_data(val_dataset)

        try:
            with open(cache_path, "wb") as f:
                logger.info(f"Saved validation cache to {cache_path}")
                pickle.dump(val_data_dict, f)
        except Exception as e:
            logger.info(f"Cache save failed: {e}")

        return val_data_dict

    def get_test_data(self, test_data_key: str) -> tp.Dict[str, jnp.ndarray]:
        """
        Checks if train data exists in the cache. If not, computes it and saves it to the cache.

        Args:
            test_data_key (str): The test data key.

        Returns:
            tp.Dict[str, jnp.ndarray]: The test data.
        """
        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, f"{test_data_key}_test")

        # TODO: fix crude workaround
        self.seed += 1_000
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded test cache from {cache_path}")
                    return pickle.load(f)
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        test_dataset = ODEDataset(self)
        test_data_dict = self.get_split_test_data(test_dataset)

        try:
            with open(cache_path, "wb") as f:
                logger.info(f"Saved test cache to {cache_path}")
                pickle.dump(test_data_dict, f)
        except Exception as e:
            logger.info(f"Cache save failed: {e}")

        return test_data_dict


class TGBDataSetCfg(pydantic.BaseModel):
    """
    Configuration class for randomly sampled disjoint train-validation-test windows on TGB temporal datasets.
    """

    name: tp.Literal["tgbn-trade", "tgbn-genre"] = pydantic.Field(
        default="tgbn-trade",
        description="Name of TGB dataset",
    )
    window_size: int = pydantic.Field(
        default=5,
        description="Number of past time steps to include in each batch",
    )
    stride: int = pydantic.Field(
        default=1,
        description="Step size for moving the window",
    )
    frequency: tp.Literal["None", "daily", "weekly", "monthly", "yearly"] = (
        pydantic.Field(
            default="None",
            description="Frequency of the time series",
        )
    )
    split_ratio: tp.List[float] = pydantic.Field(
        default=[0.6, 0.2, 0.2],
        description="Train, validation, and test ratio (sum ≤ 1).",
    )
    data_dir: str = pydantic.Field(
        default="datasets",
        description="Directory to store dataset files",
    )

    interpolation: tp.Literal["linear", "cubic"] = pydantic.Field(
        default="cubic", description="Interpolation method"
    )

    cache_dir: str = ".cache"

    seed: int = 1234

    model_config = pydantic.ConfigDict(extra="forbid")

    normalise_features: bool = pydantic.Field(
        default=False,
        description="Whether to apply a softmax normalisation over the rows of features",
    )

    # Private attributes for storing the processed snapshots and windows.
    _processed_snapshots: tp.List[Data] = pydantic.PrivateAttr()
    _train_windows: tp.List[tp.List[TemporalData]] = pydantic.PrivateAttr()
    _val_windows: tp.List[tp.List[TemporalData]] = pydantic.PrivateAttr()
    _test_windows: tp.List[tp.List[TemporalData]] = pydantic.PrivateAttr()

    _num_nodes: int = pydantic.PrivateAttr()
    _train_timestamps: tp.List[int] = pydantic.PrivateAttr()
    _val_timestamps: tp.List[int] = pydantic.PrivateAttr()
    _test_timestamps: tp.List[int] = pydantic.PrivateAttr()

    recompile: bool = False

    def _hash_config(self) -> str:
        """
        Hashes the configuration.

        Returns:
            str: The hashed configuration.
        """
        config_dict = self.model_dump(exclude={"cache_dir"})
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self, config_hash: str, data_key: str) -> Path:
        """
        Gets the cache path for the given configuration hash and data key.

        Args:
            config_hash (str): The configuration hash.
            data_key (str): The data key.

        Returns:
            Path: The cache path.
        """
        # Create cache directory if it doesn't exist
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a filename that includes both the config hash and data_key
        cache_filename = f"cache_{config_hash}_{data_key}.pkl"
        return cache_dir / cache_filename

    def __init__(self, **data):
        super().__init__(**data)

        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, "data")

        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded data cache from {cache_path}")
                    self._train_windows, self._val_windows, self._test_windows = (
                        pickle.load(f)
                    )
                    self.recompile = False
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        else:
            self._train_windows, self._val_windows, self._test_windows = (
                self.load_and_split_dataset()
            )

            # Save to cache
            try:
                with open(cache_path, "wb") as f:
                    logger.info(f"Saved data cache to {cache_path}")
                    pickle.dump(
                        (self._train_windows, self._val_windows, self._test_windows), f
                    )
                    self.recompile = True
            except Exception as e:
                logger.info(f"Cache save failed: {e}")

        logging.info(
            f"Successfully split into {len(self._train_windows)} train, {len(self._val_windows)} val, and {len(self._test_windows)} test windows."
        )

    def load_and_split_dataset(self):
        """
        Loads the dataset from TGB.
        """
        dataset = PyGNodePropPredDataset(name=self.name, root=self.data_dir)
        raw_data = dataset.get_TemporalData()

        raw_data = TemporalData(
            src=raw_data.src,
            dst=raw_data.dst,
            msg=raw_data.msg,
            t=raw_data.t,
        )

        self._num_nodes = raw_data.num_nodes
        logging.info(f"Successfully loaded dataset with {self._num_nodes} nodes.")

        # Process each snapshot (add dense adjacency matrices and node features).
        self._processed_snapshots = self.process_snapshots(raw_data)
        logging.info(
            f"Successfully processed {len(self._processed_snapshots)} snapshots."
        )

        # Split the processed snapshots into sliding windows.
        return self.sample_disjoint_windows(self._processed_snapshots)

    def process_snapshots(self, data: TemporalData) -> tp.List[Data]:
        """
        Processes the raw TemporalData snapshot-by-snapshot.

        For each unique timestamp, a snapshot is extracted, then processed to:
            - Compute a weighted adjacency matrix from the trade volume messages.
            - Set the node features to be the rows of the computed adjacency matrix.
        """

        start_time = data.t.min().item()
        end_time = data.t.max().item()

        if self.frequency == "None":
            duration = 1
        elif self.frequency == "daily":
            duration = DAY_DURATION
        elif self.frequency == "weekly":
            duration = WEEK_DURATION
        elif self.frequency == "monthly":
            duration = MONTH_DURATION
        elif self.frequency == "yearly":
            duration = YEAR_DURATION
        else:
            raise ValueError(f"Invalid frequency: {self.frequency}")

        num_snapshots = math.ceil((end_time - start_time) / duration)

        processed_snapshots = []
        current_time = start_time

        with tqdm(total=num_snapshots + 1, desc="Creating snapshots") as pbar:
            while current_time <= end_time:
                mask = (data.t >= current_time) & (data.t < (current_time + duration))
                src = data.src[mask]
                dst = data.dst[mask]
                msg = data.msg[mask].squeeze(-1) if data.msg is not None else None
                snapshot = Data(
                    src=src,
                    dst=dst,
                    edge_attributes=msg,
                    t=current_time,
                    num_nodes=self._num_nodes,
                )

                # Process snapshots
                processed_snapshot = self.process_graph_snapshot(snapshot)
                processed_snapshots.append(processed_snapshot)

                current_time += duration
                pbar.update(1)

        return processed_snapshots

    def process_graph_snapshot(self, snapshot: Data) -> Data:
        """
        Given a graph snapshot, compute:
            - A weighted adjacency matrix A where A[i,j] is the aggregate trade volume from node i to node j.
            - Node features x equal to the rows of A.
        The snapshot is updated with two new attributes: 'adj' and 'x'.
        """
        # Use the torch API to generate a dense adjacency matrix from the edge list
        adj = to_dense_adj(
            edge_index=torch.stack([snapshot.src, snapshot.dst], dim=0),
            edge_attr=snapshot.edge_attributes,
            max_num_nodes=self._num_nodes,
        ).squeeze(0)

        # Compute node features as the rows of the adjacency matrix
        x = adj
        # x = torch.nn.functional.softmax(adj, dim=-1)

        # Update the snapshot with the new attributes
        snapshot.adj = adj
        snapshot.x = x

        return snapshot

    def sample_disjoint_windows(self, snapshots: tp.List[Data]):
        """
        Randomly samples disjoint train/validation/test windows from the dataset.
        Ensures no overlap between them.
        """
        num_snapshots = len(snapshots)

        # Generate all possible window start indices
        window_starts = np.arange(0, num_snapshots - self.window_size + 1, self.stride)
        np.random.shuffle(window_starts)  # Randomize order

        # Determine split sizes
        num_train = int(len(window_starts) * self.split_ratio[0])
        num_val = int(len(window_starts) * self.split_ratio[1])
        num_test = len(window_starts) - num_train - num_val

        # Select random, disjoint windows for train, validation, and test
        train_starts = window_starts[:num_train]
        val_starts = window_starts[num_train : num_train + num_val]
        test_starts = window_starts[
            num_train + num_val : num_train + num_val + num_test
        ]
        logging.info("Train starts: {}".format(train_starts))
        logging.info("Validation starts: {}".format(val_starts))
        logging.info("Test starts: {}".format(test_starts))

        # Convert window starts to actual timestamps
        train_windows = [
            snapshots[start : start + self.window_size] for start in train_starts
        ]
        val_windows = [
            snapshots[start : start + self.window_size] for start in val_starts
        ]
        test_windows = [
            snapshots[start : start + self.window_size] for start in test_starts
        ]

        # Flatten lists of timestamps
        self._train_timestamps = torch.tensor(
            [snapshot.t for window in train_windows for snapshot in window]
        )
        self._val_timestamps = torch.tensor(
            [snapshot.t for window in val_windows for snapshot in window]
        )
        self._test_timestamps = torch.tensor(
            [snapshot.t for window in test_windows for snapshot in window]
        )

        return train_windows, val_windows, test_windows

    def get_interpolation_coeffs(
        self, ts: jnp.ndarray, signal: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Stacks signal matrices with time stamps and gets the graph interpolation coefficients.

        Args:
            ts (jnp.ndarray): The time stamps.
            signal (jnp.ndarray): The signal matrices.

        Returns:
            jnp.ndarray: The interpolation coefficients.
        """
        t_index = jnp.broadcast_to(
            ts[:, None, None],
            (ts.shape[0], signal.shape[1], signal.shape[2]),
        )

        X = jnp.stack([t_index, signal], axis=-1)

        if self.interpolation == "linear":
            coeffs = diffrax.linear_interpolation(ts, X)
        elif self.interpolation == "cubic":
            coeffs = diffrax.backward_hermite_coefficients(ts, X)
            coeffs = [torch.tensor(np.array(coeff)) for coeff in coeffs]
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

        return coeffs

    def process_window(self, window: tp.List[Data]) -> tp.Dict:
        # delete the last snapshot from windows
        last_snapshot = window[-1]
        window = window[:-1]

        times = torch.arange(len(window))

        self._num_nodes = last_snapshot.x.shape[0]
        row_indices = torch.arange(self._num_nodes)
        source_mask = torch.isin(row_indices, last_snapshot.src)

        if self.normalise_features:
            x_t = torch.stack(
                [torch.nn.functional.softmax(snapshot.x, dim=-1) for snapshot in window]
            )
        else:
            x_t = torch.stack([snapshot.x for snapshot in window])

        data_dict = {
            "t": times,
            "A": torch.stack([snapshot.adj for snapshot in window]),
            "A_t": times,
            "events_time": times,
            "true_y": last_snapshot.x,
            "true_y0": window[0].x,
            "x_t": x_t,
            "source_mask": source_mask,
            "start_time": window[0].t,
        }

        # get interpolation coeficients
        data_dict["graph_path_coeffs"] = self.get_interpolation_coeffs(
            jnp.array(times), jnp.array(data_dict["A"])
        )
        data_dict["x_coeffs"] = self.get_interpolation_coeffs(
            jnp.array(times),
            jnp.array(data_dict["x_t"]),
        )

        return data_dict

    def process_windows(self, windows: tp.List[tp.List[Data]]) -> tp.List[tp.Dict]:
        # Apply process_window to each window
        return [self.process_window(window) for window in windows]

    def get_training_data(self, train_data_key) -> SlidingWindowTemporalLoader:
        """
        Returns a SlidingWindowTemporalLoader for the training windows.
        Each batch is a list of processed TemporalData snapshots (one window).
        """
        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, f"{train_data_key}_train")

        # Try to load from cache
        if cache_path.exists() and not self.recompile:
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded training cache from {cache_path}")
                    train_dict = pickle.load(f)
                    return SlidingWindowTemporalLoader(train_dict, batch_size=1)
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        train_dict = self.process_windows(self._train_windows)

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                logger.info(f"Saved training cache to {cache_path}")
                pickle.dump(train_dict, f)
        except Exception as e:
            logger.info(f"Cache save failed: {e}")

        return SlidingWindowTemporalLoader(train_dict, batch_size=1)

    def get_validation_data(self, val_data_key) -> SlidingWindowTemporalLoader:
        """
        Returns a SlidingWindowTemporalLoader for the validation windows.
        """
        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, f"{val_data_key}_val")

        # Try to load from cache
        if cache_path.exists() and not self.recompile:
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded validation cache from {cache_path}")
                    val_dict = pickle.load(f)
                    return SlidingWindowTemporalLoader(val_dict, batch_size=1)
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        val_dict = self.process_windows(self._val_windows)

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                logger.info(f"Saved validation cache to {cache_path}")
                pickle.dump(val_dict, f)
        except Exception as e:
            logger.info(f"Cache save failed: {e}")

        return SlidingWindowTemporalLoader(val_dict, batch_size=1)

    def get_test_data(self, test_data_key) -> SlidingWindowTemporalLoader:
        """
        Returns a SlidingWindowTemporalLoader for the testing windows.
        """
        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, f"{test_data_key}_test")

        # Try to load from cache
        if cache_path.exists() and not self.recompile:
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded test cache from {cache_path}")
                    test_dict = pickle.load(f)
                    return SlidingWindowTemporalLoader(test_dict, batch_size=1)
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        test_dict = self.process_windows(self._test_windows)

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                logger.info(f"Saved test cache to {cache_path}")
                pickle.dump(test_dict, f)
        except Exception as e:
            logger.info(f"Cache save failed: {e}")

        return SlidingWindowTemporalLoader(test_dict, batch_size=1)


class PGTDataSetCfg(pydantic.BaseModel):
    """
    Configuration class for randomly sampled disjoint train-validation-test windows on Pytroch Geometric Temporal datasets.
    """

    name: tp.Literal["england-covid", "twitter-tennis"] = pydantic.Field(
        ...,
        description="Name of PGT dataset",
    )
    window_size: int = pydantic.Field(
        default=5,
        description="Number of past time steps to include in each batch",
    )
    stride: int = pydantic.Field(
        default=1,
        description="Step size for moving the window",
    )
    split_ratio: tp.List[float] = pydantic.Field(
        default=[0.6, 0.2, 0.2],
        description="Train, validation, and test ratio (sum ≤ 1).",
    )

    interpolation: tp.Literal["linear", "cubic"] = pydantic.Field(
        default="cubic", description="Interpolation method"
    )

    cache_dir: str = ".cache"

    seed: int = 1234

    model_config = pydantic.ConfigDict(extra="forbid")

    # Private attributes for storing the processed snapshots and windows.
    _processed_snapshots: tp.List[Data] = pydantic.PrivateAttr()
    _train_windows: tp.List[tp.List[TemporalData]] = pydantic.PrivateAttr()
    _val_windows: tp.List[tp.List[TemporalData]] = pydantic.PrivateAttr()
    _test_windows: tp.List[tp.List[TemporalData]] = pydantic.PrivateAttr()

    _num_nodes: int = pydantic.PrivateAttr()

    recompile: bool = False

    def _hash_config(self) -> str:
        """
        Hashes the configuration.

        Returns:
            str: The hashed configuration.
        """
        config_dict = self.model_dump(exclude={"cache_dir"})
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self, config_hash: str, data_key: str) -> Path:
        """
        Gets the cache path for the given configuration hash and data key.

        Args:
            config_hash (str): The configuration hash.
            data_key (str): The data key.

        Returns:
            Path: The cache path.
        """
        # Create cache directory if it doesn't exist
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a filename that includes both the config hash and data_key
        cache_filename = f"cache_{config_hash}_{data_key}.pkl"
        return cache_dir / cache_filename

    def __init__(self, **data):
        super().__init__(**data)

        if self.name == "england-covid":
            cache_path = ".datasets/england_dataset.pkl"
        elif self.name == "twitter-tennis":
            cache_path = ".datasets/twitter_dataset.pkl"
        else:
            raise ValueError(f"Unknown dataset {self.name}")

        try:
            with open(cache_path, "rb") as f:
                logger.info(f"Loaded data cache from {cache_path}")
                dataset = pickle.load(f)
        except Exception as e:
            logger.info(f"Cache load failed: {e}. Computing fresh result.")

        self._processed_snapshots = self.process_snapshots(dataset)

        self._train_windows, self._val_windows, self._test_windows = (
            self.sample_disjoint_windows(self._processed_snapshots)
        )

        logging.info(
            f"Successfully split into {len(self._train_windows)} train, {len(self._val_windows)} val, and {len(self._test_windows)} test windows."
        )

    def process_snapshots(self, data: tp.List[Data]) -> tp.List[Data]:
        """
        Processes the raw TemporalData snapshot-by-snapshot.

        For each unique timestamp, a snapshot is extracted, then processed to:
            - Compute a weighted adjacency matrix from the trade volume messages.
            - Set the node features to be the rows of the computed adjacency matrix.
        """
        return [self.process_graph_snapshot(snapshot) for snapshot in data]

    def process_graph_snapshot(self, snapshot: Data) -> Data:
        """
        Given a graph snapshot, compute:
            - A weighted adjacency matrix A where A[i,j] is the aggregate trade volume from node i to node j.
            - Node features x equal to the rows of A.
        The snapshot is updated with two new attributes: 'adj' and 'x'.
        """
        # Use the torch API to generate a dense adjacency matrix from the edge list
        adj = to_dense_adj(
            edge_index=snapshot.edge_index,
            edge_attr=snapshot.edge_attr,
            max_num_nodes=snapshot.num_nodes,
        ).squeeze(0)

        # Update the snapshot with the new attributes
        snapshot.adj = adj

        return snapshot

    def sample_disjoint_windows(self, snapshots: tp.List[Data]):
        """
        Randomly samples disjoint train/validation/test windows from the dataset.
        Ensures no overlap between them.
        """
        num_snapshots = len(snapshots)

        # Generate all possible window start indices
        window_starts = np.arange(0, num_snapshots - self.window_size + 1, self.stride)
        np.random.shuffle(window_starts)  # Randomize order

        # Determine split sizes
        num_train = int(len(window_starts) * self.split_ratio[0])
        num_val = int(len(window_starts) * self.split_ratio[1])
        num_test = len(window_starts) - num_train - num_val

        # Select random, disjoint windows for train, validation, and test
        train_starts = window_starts[:num_train]
        val_starts = window_starts[num_train : num_train + num_val]
        test_starts = window_starts[
            num_train + num_val : num_train + num_val + num_test
        ]
        logging.info("Train starts: {}".format(train_starts))
        logging.info("Validation starts: {}".format(val_starts))
        logging.info("Test starts: {}".format(test_starts))

        # Convert window starts to actual timestamps
        train_windows = [
            snapshots[start : start + self.window_size] for start in train_starts
        ]
        val_windows = [
            snapshots[start : start + self.window_size] for start in val_starts
        ]
        test_windows = [
            snapshots[start : start + self.window_size] for start in test_starts
        ]

        return train_windows, val_windows, test_windows

    def get_interpolation_coeffs(
        self, ts: jnp.ndarray, signal: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Stacks signal matrices with time stamps and gets the graph interpolation coefficients.

        Args:
            ts (jnp.ndarray): The time stamps.
            signal (jnp.ndarray): The signal matrices.

        Returns:
            jnp.ndarray: The interpolation coefficients.
        """
        t_index = jnp.broadcast_to(
            ts[:, None, None],
            (ts.shape[0], signal.shape[1], signal.shape[2]),
        )

        X = jnp.stack([t_index, signal], axis=-1)

        if self.interpolation == "linear":
            coeffs = diffrax.linear_interpolation(ts, X)
        elif self.interpolation == "cubic":
            coeffs = diffrax.backward_hermite_coefficients(ts, X)
            coeffs = [torch.tensor(np.array(coeff)) for coeff in coeffs]
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

        return coeffs

    def process_window(self, window: tp.List[Data]) -> tp.Dict:
        # delete the last snapshot from windows
        last_snapshot = window[-1]
        window = window[:-1]

        times = torch.arange(len(window))

        x_t = torch.stack([snapshot.x for snapshot in window])

        data_dict = {
            "t": times,
            "A": torch.stack([snapshot.adj for snapshot in window]),
            "A_t": times,
            "events_time": times,
            "true_y": last_snapshot.y,
            "true_y0": window[0].x,
            "x_t": x_t,
        }

        # get interpolation coeficients
        data_dict["graph_path_coeffs"] = self.get_interpolation_coeffs(
            jnp.array(times), jnp.array(data_dict["A"])
        )
        data_dict["x_coeffs"] = self.get_interpolation_coeffs(
            jnp.array(times),
            jnp.array(data_dict["x_t"]),
        )

        return data_dict

    def process_windows(self, windows: tp.List[tp.List[Data]]) -> tp.List[tp.Dict]:
        # Apply process_window to each window
        return [self.process_window(window) for window in windows]

    def get_training_data(self, train_data_key) -> SlidingWindowTemporalLoader:
        """
        Returns a SlidingWindowTemporalLoader for the training windows.
        Each batch is a list of processed TemporalData snapshots (one window).
        """
        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, f"{train_data_key}_train")

        # Try to load from cache
        if cache_path.exists() and not self.recompile:
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded training cache from {cache_path}")
                    train_dict = pickle.load(f)
                    return SlidingWindowTemporalLoader(train_dict, batch_size=1)
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        train_dict = self.process_windows(self._train_windows)

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                logger.info(f"Saved training cache to {cache_path}")
                pickle.dump(train_dict, f)
        except Exception as e:
            logger.info(f"Cache save failed: {e}")

        return SlidingWindowTemporalLoader(train_dict, batch_size=1)

    def get_validation_data(self, val_data_key) -> SlidingWindowTemporalLoader:
        """
        Returns a SlidingWindowTemporalLoader for the validation windows.
        """
        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, f"{val_data_key}_val")

        # Try to load from cache
        if cache_path.exists() and not self.recompile:
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded validation cache from {cache_path}")
                    val_dict = pickle.load(f)
                    return SlidingWindowTemporalLoader(val_dict, batch_size=1)
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        val_dict = self.process_windows(self._val_windows)

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                logger.info(f"Saved validation cache to {cache_path}")
                pickle.dump(val_dict, f)
        except Exception as e:
            logger.info(f"Cache save failed: {e}")

        return SlidingWindowTemporalLoader(val_dict, batch_size=1)

    def get_test_data(self, test_data_key) -> SlidingWindowTemporalLoader:
        """
        Returns a SlidingWindowTemporalLoader for the testing windows.
        """
        config_hash = self._hash_config()
        cache_path = self._get_cache_path(config_hash, f"{test_data_key}_test")

        # Try to load from cache
        if cache_path.exists() and not self.recompile:
            try:
                with open(cache_path, "rb") as f:
                    logger.info(f"Loaded test cache from {cache_path}")
                    test_dict = pickle.load(f)
                    return SlidingWindowTemporalLoader(test_dict, batch_size=1)
            except Exception as e:
                logger.info(f"Cache load failed: {e}. Computing fresh result.")

        test_dict = self.process_windows(self._test_windows)

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                logger.info(f"Saved test cache to {cache_path}")
                pickle.dump(test_dict, f)
        except Exception as e:
            logger.info(f"Cache save failed: {e}")

        return SlidingWindowTemporalLoader(test_dict, batch_size=1)
