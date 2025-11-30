import logging

import typing as tp
import numpy as np
import networkx as nx
import jax
import jax.numpy as jnp
import diffrax
from .data_tools import *
from .ode_models import *
from .misc import sample_non_overlapping_rect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ODEDataset(object):
    """
    ODEDataset class
    """

    def __init__(self, cfg):
        self.data_name = cfg.name

        if cfg.name == "heat":
            self.ode_model = HeatDiffusion
        elif cfg.name == "gene":
            self.ode_model = GeneDynamics
        elif cfg.name == "mutualistic":
            self.ode_model = MutualDynamics
        elif cfg.name == "sir":
            self.ode_model = SIRDynamics
        else:
            raise ValueError("Data name {} is not supported".format(cfg.data_name))

        self.num_nodes = cfg.num_nodes
        self.split_ratio = cfg.split_ratio
        self.T = cfg.final_time
        self.time_tick = cfg.time_tick
        self.graph_type = cfg.graph_type
        self.operator_type = cfg.operator_type
        self.sampling_type = cfg.sampling_type
        self.layout = cfg.layout
        self.ode_solver = cfg.method
        self.dt0 = cfg.dt0
        self.enable_dynamic_graph = cfg.dynamic_graph
        self.enable_all_dynamic = cfg.all_dynamic
        self.seed = cfg.seed
        self.batch_size = cfg.batch_size
        self.amp_range = cfg.amp_range

        self.beta = cfg.beta
        self.gamma = cfg.gamma

        self.N = int(np.ceil(np.sqrt(self.num_nodes)))  # grid-layout pixels :20
        self.sparse_scale = 10
        self.event_times = 10  # the number of graph structure changes
        self.event_p = 0.001  # the possibility of graph structure changes
        if self.graph_type == "mixed":
            self.labels = np.repeat(np.arange(3), self.batch_size).reshape(-1, 1)
        else:
            self.labels = np.concatenate(
                [np.zeros(self.batch_size), np.ones(self.batch_size)]
            )

        # gen data
        self.A, self.D, self.L = None, None, None
        self.events_time, self.events_time_indices = None, None
        self.refine_event_times()
        self.t = (
            self.gen_sampling_time()
        )  # sample time stamps based on self.sampling_type
        self.initialize()  # initial state; shape = (num_nodes, 1)
        self.true_y = (
            self.gen_all_data()
        )  # generate the dynamic data, shape = (num_nodes, time_tick)

    def refine_event_times(self):
        """
        Adjust the number of event times based on the split ratio and constraints.

        If `enable_all_dynamic` is True, the number of event times is increased by a factor
        derived from the split ratio.

        This method modifies the `event_times` attribute of the class.
        """
        if self.enable_all_dynamic:
            self.event_times += int(
                self.event_times / self.split_ratio[0] * self.split_ratio[1]
            )

    def initialize(self) -> np.ndarray:
        """
        Create a batch of grids with spatially isolated center patches.

        Each grid is of size (N x N) and will contain three non-overlapping
        patches (centers) with randomized positions and amplitudes. The patch sizes
        are fixed relative to N (based on your original fractions), and the amplitudes
        are sampled uniformly from amp_range.

        After initialization, self.x0 is set to an array of shape (batch_size, N*N, 1).

        Parameters:
            batch_size (int): number of grids (batch items) to generate.
            amp_range (tuple): (min_amplitude, max_amplitude) range for random amplitudes.
        """
        self.x0 = np.zeros((self.batch_size, self.N, self.N))

        # --- Define the relative sizes for each center patch ---
        # (height_fraction, width_fraction)
        # These are the same as in your original code:
        #   Center 1: ~20% of N by 20% of N,
        #   Center 2: ~30% of N by 30% of N,
        #   Center 3: ~20% of N by 30% of N.
        center_sizes = [(0.20, 0.20), (0.30, 0.30), (0.20, 0.30)]

        # --- For each grid in the batch, sample the centers ---
        for i in range(self.x0.shape[0]):
            existing_rects = (
                []
            )  # to keep track of already placed patches (for non-overlap)
            for frac_h, frac_w in center_sizes:
                # Compute patch size in pixels (ensure at least 1 pixel in each dimension)
                h = max(1, int(frac_h * self.N))
                w = max(1, int(frac_w * self.N))
                # Sample a non-overlapping rectangle for this patch
                r1, c1, r2, c2 = sample_non_overlapping_rect(
                    self.N, h, w, existing_rects
                )
                existing_rects.append((r1, c1, r2, c2))
                # Randomize amplitude for this patch
                amp = np.random.uniform(*self.amp_range)
                # Set the patch region in the grid to the chosen amplitude
                self.x0[i, r1:r2, c1:c2] = amp

        # Flatten each grid into a column vector (so each row of self.x0 corresponds to one grid)
        self.x0 = self.x0.reshape(self.x0.shape[0], -1, 1)

        return self.x0

    def _gen_grid_graph(self):
        # Assumes grid_8_neighbor_graph builds a grid graph with 8-neighbor connectivity.
        A = grid_8_neighbor_graph(self.N)
        A = np.tile(A[None, ...], (self.batch_size, 1, 1))
        return A

    def _gen_random_graph(self):
        A_list = []
        for i in range(self.batch_size):
            # Varying the seed (or using additional randomness) to generate different graphs.
            G = nx.erdos_renyi_graph(self.num_nodes, 0.1, seed=self.seed + i)
            G = networkx_reorder_nodes(G, self.layout)
            A_list.append(np.array(nx.to_numpy_array(G)))
        A = np.stack(A_list, axis=0)
        return A

    def _gen_power_law_graph(self):
        A_list = []
        for i in range(self.batch_size):
            G = nx.barabasi_albert_graph(self.num_nodes, 5, seed=self.seed + i)
            G = networkx_reorder_nodes(G, self.layout)
            A_list.append(np.array(nx.to_numpy_array(G), dtype=float))
        A = np.stack(A_list, axis=0)
        return A

    def _gen_small_world_graph(self):
        A_list = []
        for i in range(self.batch_size):
            G = nx.newman_watts_strogatz_graph(
                self.num_nodes, 5, 0.5, seed=self.seed + i
            )
            G = networkx_reorder_nodes(G, self.layout)
            A_list.append(np.array(nx.to_numpy_array(G), dtype=float))
        A = np.stack(A_list, axis=0)
        return A

    def _gen_scale_free_graph(self):
        A_list = []
        for i in range(self.batch_size):
            G = nx.scale_free_graph(
                self.num_nodes, alpha=0.5, beta=0.5, gamma=0.5, seed=self.seed + i
            )
            G = networkx_reorder_nodes(G, self.layout)
            A_list.append(np.array(nx.to_numpy_array(G), dtype=float))
        A = np.stack(A_list, axis=0)
        return A

    def _gen_community_graph(self):
        A_list = []
        for i in range(self.batch_size):
            n1 = int(self.num_nodes / 3)
            n2 = int(self.num_nodes / 3)
            n3 = int(self.num_nodes / 4)
            n4 = self.num_nodes - n1 - n2 - n3
            G = nx.random_partition_graph(
                [n1, n2, n3, n4], 0.25, 0.01, seed=self.seed + i
            )
            G = networkx_reorder_nodes(G, self.layout)
            A_list.append(np.array(nx.to_numpy_array(G), dtype=float))
        A = np.stack(A_list, axis=0)
        return A

    def gen_static_graph(self):
        """
        Generate a static graph based on self.graph_type and self.num_nodes.

        Returns:
            A (np.ndarray): The adjacency matrix of the graph.
            D (np.ndarray): The degree matrix of the graph.
            L (np.ndarray): The Laplacian matrix of the graph (D - A).
        """
        if self.graph_type == "grid":
            A = self._gen_grid_graph()
        elif self.graph_type == "random":
            A = self._gen_random_graph()
        elif self.graph_type == "power_law":
            A = self._gen_power_law_graph()
        elif self.graph_type == "small_world":
            A = self._gen_small_world_graph()
        elif self.graph_type == "community":
            A = self._gen_community_graph()
        elif self.graph_type == "mixed":
            A = np.concatenate(
                [
                    self._gen_grid_graph(),
                    # self._gen_random_graph(),
                    self._gen_power_law_graph(),
                    # self._gen_small_world_graph(),
                    self._gen_community_graph(),
                ],
                axis=0,
            )
        else:
            raise ValueError(f"Graph type {self.graph_type} is not supported")

        # Compute the degree and Laplacian matrices for each graph in the batch.
        D_list = []
        L_list = []
        for i in range(self.batch_size):
            A_i = A[i]
            D_i = np.diag(A_i.sum(axis=1))
            L_i = D_i - A_i
            D_list.append(D_i)
            L_list.append(L_i)
        D = np.stack(D_list, axis=0)
        L = np.stack(L_list, axis=0)

        return A, D, L

    def gen_static_graph_data(
        self,
        L: np.ndarray,
        A: np.ndarray,
        x0: np.ndarray,
        t: np.ndarray,
        method: str = "Dopri5",
    ) -> np.ndarray:
        """
        Solve the ODE defined by self.ode_model for a batched initial condition x0,
        using a batched time grid t. This function vectorizes the ODE solve over the batch dimension.
        """

        def solve_ode_single(
            x0_single, t_single, L_single, A_single, beta=0.0, gamma=0.0
        ):
            # Choose the correct ODE function based on the data name.
            if self.data_name.lower() == "heat":
                ode_fn = self.ode_model(L_single, 1)
            elif self.data_name.lower() == "gene":
                ode_fn = self.ode_model(A_single, 1)
            elif self.data_name.lower() == "mutualistic":
                ode_fn = self.ode_model(A_single)
            elif self.data_name.lower() == "sir":
                ode_fn = self.ode_model(A_single, gamma, beta)
            else:
                raise ValueError(f"ODE Type: {self.data_name} is not supported")

            def f_diffrax(t_val, y, args):
                return ode_fn(t_val, y)

            term = diffrax.ODETerm(f_diffrax)
            solver = getattr(diffrax, self.ode_solver)()
            y0_single = jnp.array(x0_single)
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=t_single[0],
                t1=t_single[-1],
                dt0=self.dt0,
                y0=y0_single,
                saveat=diffrax.SaveAt(ts=t_single),
            )
            # sol.ys is an array of shape (num_time_points, num_nodes, 1)
            return sol.ys

        sol_batched = jax.vmap(solve_ode_single, in_axes=(0, 0, 0, 0))(x0, t, L, A)
        # sol_batched has shape (batch_size, num_time_points, num_nodes, 1);
        # swap axes so that time is the first axis.
        sol_batched = jnp.swapaxes(sol_batched, 0, 1)
        return np.array(sol_batched)

    def gen_sampling_time(self) -> np.ndarray:
        """
        Generate time sampling points based on the specified sampling type.

        Returns:
            np.ndarray: A 2D array of shape (batch_size, num_points) containing the time sampling points.
        """
        if self.sampling_type == "equal":
            logger.info("Build Equally-sampled time dynamics")
            # Generate equally spaced time ticks\
            t = np.linspace(0.0, self.T, self.time_tick)
            # Tile the same time vector for each batch instance
            return np.tile(t, (self.batch_size, 1))
        elif self.sampling_type == "irregular":
            logger.info("Building irregularly-sampled time dynamics")
            # Create a full set of equally spaced ticks
            t_full = np.linspace(0.0, self.T, self.time_tick * self.sparse_scale)
            # Randomly select  1.2 * time_tick time points and sort them
            # TODO: make 1.2 a hyperparemeter
            num_points = int(
                self.time_tick * 1.2
            )  # Consider making 1.2 a hyperparameter

            # Initialize a list to hold each batch row
            t_batch = []

            bs = self.batch_size
            for _ in range(bs):
                # Randomly permute t_full, take the first num_points,
                # then sort them to maintain time order
                t_shuffled = np.random.permutation(t_full)[:num_points]
                t_shuffled = np.sort(t_shuffled)
                # Force the first time point to be 0
                t_shuffled[0] = 0.0
                t_batch.append(t_shuffled)

            # Stack the list into a 2D NumPy array of shape (batch_size, num_points)
            return np.stack(t_batch, axis=0)
        else:
            raise ValueError(f"{self.sampling_type} sampling is not supported")

    def split_train_val_test(
        self,
    ) -> tp.Tuple[tp.List[int], tp.List[int], tp.Optional[tp.List[int]]]:
        """
        Split the dataset into training, interpolation testing, and extrapolation testing sets.

        This method determines the indices for the training set, interpolation testing set,
        and extrapolation testing set based on the sampling type and split ratios.

        Returns:
            tp.Tuple[tp.List[int], tp.List[int], tp.Optional[tp.List[int]]]:
                A tuple containing:
                - id_train: List of indices for the training set.
                - id_test_extra: List of indices for the extrapolation testing set.
                - id_test_inter: List of indices for the interpolation testing set (or None if not applicable).
        """
        if self.sampling_type == "equal":
            # first 80 % for train
            id_train = list(range(round(self.time_tick * self.split_ratio[0])))
            # last 20 % for test (extrapolation)
            id_test_extra = list(
                range(round(self.time_tick * self.split_ratio[0]), self.time_tick)
            )
            id_test_inter = None
        elif self.sampling_type == "irregular":
            # For irregular sampling: the last points beyond time_tick for extrapolation testing.
            id_test_extra = list(
                range(
                    self.time_tick, round(self.time_tick * (1.0 + self.split_ratio[1]))
                )
            )
            # Randomly select some points from within the first time_tick for interpolation testing.
            all_indices = list(range(1, self.time_tick))
            id_test_inter = np.random.permutation(all_indices)[
                : round(self.time_tick * self.split_ratio[1])
            ].tolist()
            id_test_inter.sort()
            id_train = list(sorted(set(range(self.time_tick)) - set(id_test_inter)))
        else:
            raise ValueError(
                "{} sampling manner is not supported".format(self.sampling_type)
            )
        return id_train, id_test_extra, id_test_inter

    def gen_all_data(self):
        """
        Generate the full dynamic dataset.

        When the graph is static, a single call is made that is vectorized over the batch.
        When the graph is dynamic (i.e. changes at a number of event times), we use a
        JAX lax.scan to loop over events—each event simulation is vectorized over the batch.
        """
        # --- Static Graph Case ---
        if not self.enable_dynamic_graph:
            A, D, L = self.gen_static_graph()  # Generate a single static graph.
            solution_numerical = self.gen_static_graph_data(
                L, A, self.x0, self.t, self.ode_solver
            )
            self.A, self.D, self.L = A, D, L

        # --- Dynamic Graph Case ---
        else:
            # Generate the static graph (used as a baseline for dynamic events)
            A, D, L = self.gen_static_graph()

            # Generate events per batch element.
            # Here, self.t is assumed to have shape (batch_size, time_steps).
            events_time, event_indices = gen_events_happen_time(
                self.t, self.event_times, self.split_ratio, self.enable_all_dynamic
            )
            logger.info(
                "Final time: {},\nBatched event times shape:{}".format(
                    self.T, events_time.shape
                )
            )
            logger.info(
                "t shape: {},\nEvent indices:{}".format(self.t.shape, event_indices)
            )

            A_list, D_list, L_list = gen_events_happen_graph(
                A, self.event_times, self.event_p
            )  # get list of adjacencies, D's and L's (of length self.events_time) that each time an event happends randomly drops or adds some edges

            solution_numerical = []
            for event_idx in range(len(A_list)):
                if event_idx == 0:
                    pre_solution_numerical = self.gen_static_graph_data(
                        L_list[event_idx],
                        A_list[event_idx],
                        self.x0,
                        self.t[:, : event_indices[event_idx]],
                        self.ode_solver,
                    )
                elif event_idx == self.event_times:
                    pre_solution_numerical = self.gen_static_graph_data(
                        L_list[event_idx],
                        A_list[event_idx],
                        pre_solution_numerical[-1],
                        self.t[:, event_indices[event_idx - 1] :],
                        self.ode_solver,
                    )
                else:
                    pre_solution_numerical = self.gen_static_graph_data(
                        L_list[event_idx],
                        A_list[event_idx],
                        pre_solution_numerical[-1],
                        self.t[
                            :, event_indices[event_idx - 1] : event_indices[event_idx]
                        ],
                        self.ode_solver,
                    )

                solution_numerical.append(pre_solution_numerical)
            solution_numerical = np.concatenate(
                solution_numerical, axis=0
            )  # [120, 64, 400, 1]

            # Update dataset params
            self.A, self.D, self.L = (
                np.transpose(np.array(A_list), (1, 0, 2, 3)),
                np.transpose(np.array(D_list), (1, 0, 2, 3)),
                np.transpose(np.array(L_list), (1, 0, 2, 3)),
            )
            self.events_time, self.events_time_indices = events_time, event_indices

        # self.vis(solution_numerical)
        solution_numerical = np.transpose(
            np.squeeze(solution_numerical, axis=-1), (1, 0, 2)
        )

        return solution_numerical
