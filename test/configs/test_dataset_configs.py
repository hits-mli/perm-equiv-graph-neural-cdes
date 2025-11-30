import pytest

import torch
import torch.testing
from torch_geometric.data import Data, TemporalData

from configs.dataset_configs import TGBDataSetCfg


@pytest.fixture
def mock_data():
    """Fixture for mocked Data."""
    # Create a mock Data object
    src = torch.tensor(list(range(30)))
    dst = torch.tensor(list(range(30))) + 1
    msg = torch.ones((30,))
    t = torch.tensor(1.0)
    return Data(edge_index=torch.stack([src, dst], dim=0), edge_attributes=msg, t=t)


@pytest.fixture
def mock_temporal_data():
    """Fixture for mocked TemporalData."""
    # Create a mock TemporalData object
    t = torch.tensor(list(range(30))) + 1
    src = torch.tensor(list(range(30)))
    dst = torch.tensor(list(range(30))) + 1
    msg = torch.ones((30, 1))
    return TemporalData(src=src, dst=dst, msg=msg, t=t)


@pytest.fixture
def config(mock_temporal_data, mocker):
    """Fixture to instantiate the configuration object."""
    # Mock the load_dataset method to return the mock_data
    mocker.patch.object(TGBDataSetCfg, "load_dataset", return_value=mock_temporal_data)

    # Instantiate and return the configuration object
    return TGBDataSetCfg(
        name="tgbn-trade",
        window_size=3,
        stride=3,
        split_ratio=[0.6, 0.2, 0.2],
        data_dir="datasets",
        seed=42,
    )


@pytest.fixture
def snapshots_actual(mocker):
    """Fixture to instantiate the configuration object."""

    # Instantiate and return the configuration object
    return TGBDataSetCfg(
        name="tgbn-trade",
        window_size=3,
        stride=3,
        split_ratio=[0.6, 0.2, 0.2],
        data_dir="datasets",
        seed=42,
    )._processed_snapshots


def test_process_snapshots(config, mock_temporal_data, mocker):
    """Test the processing of snapshots."""

    # Reinitialise to process the mock data
    config.__init__(
        name="tgbn-trade",
        window_size=3,
        stride=3,
        split_ratio=[0.6, 0.2, 0.2],
        data_dir="datasets",
        seed=42,
    )

    # Access the processed snapshots
    processed_snapshots = config._processed_snapshots

    # Assertions
    assert len(processed_snapshots) == 30  # Assuming one snapshot per unique timestamp
    for snapshot in processed_snapshots:
        assert snapshot.t in mock_temporal_data.t
        assert hasattr(snapshot, "adj")  # Check if 'adj' attribute is present
        assert hasattr(snapshot, "x")  # Check if 'x' attribute is present
        assert snapshot.adj.shape[0] == config._num_nodes  # Adjacency matrix size
        assert snapshot.adj.shape[1] == config._num_nodes  # Adjacency matrix size
        assert snapshot.x.shape[0] == config._num_nodes  # Node feature size
        assert snapshot.x.shape[1] == config._num_nodes  # Node feature size


def test_sample_disjoint_windows_mock(config, mock_data):
    """Test the sample_disjoint_windows method with mock data."""
    # Prepare the snapshots (mock data list)
    snapshots = [mock_data] * 30  # 30 snapshots

    # Call the method under test
    train_windows, val_windows, test_windows = config.sample_disjoint_windows(snapshots)

    # Check the sizes of the splits based on the split_ratio
    num_snapshots = len(snapshots)
    num_windows = len(train_windows) + len(val_windows) + len(test_windows)

    # Assert that the total number of windows is equal to the possible windows
    num_possible_windows = (num_snapshots - config.window_size) // config.stride + 1
    assert num_windows == num_possible_windows

    # Assert the split ratio correctness
    num_train = int(num_possible_windows * config.split_ratio[0])
    num_val = int(num_possible_windows * config.split_ratio[1])
    num_test = num_possible_windows - num_train - num_val

    assert len(train_windows) == num_train
    assert len(val_windows) == num_val
    assert len(test_windows) == num_test


def test_sample_disjoint_windows_actual(config, snapshots_actual):
    """Test the sample_disjoint_windows method with actual data."""
    # Call the method under test
    train_windows, val_windows, test_windows = config.sample_disjoint_windows(
        snapshots_actual
    )

    # Check the sizes of the splits based on the split_ratio
    num_snapshots = len(snapshots_actual)
    num_windows = len(train_windows) + len(val_windows) + len(test_windows)

    # Assert that the total number of windows is equal to the possible windows
    num_possible_windows = (num_snapshots - config.window_size) // config.stride + 1
    assert num_windows == num_possible_windows

    # Assert the split ratio correctness
    num_train = int(num_possible_windows * config.split_ratio[0])
    num_val = int(num_possible_windows * config.split_ratio[1])
    num_test = num_possible_windows - num_train - num_val

    assert len(train_windows) == num_train
    assert len(val_windows) == num_val
    assert len(test_windows) == num_test

    # Ensure the windows are disjoint (no overlap between train, val, test)
    all_windows = train_windows + val_windows + test_windows
    flattened_windows = [snapshot.t for window in all_windows for snapshot in window]

    # Check that no timestamp is repeated across any window sets
    assert len(flattened_windows) == len(set(flattened_windows))

    # Check that the window size is respected (each window should have exactly `window_size` snapshots)
    for window in all_windows:
        assert len(window) == config.window_size

    # Ensure that the timestamps are ordered correctly within each window
    for window in all_windows:
        timestamps = [snapshot.t for snapshot in window]
        assert sorted(timestamps) == timestamps


def test_get_train_loader(config, mock_temporal_data):
    train_loader = config.get_training_data()
    assert torch.equal(next(iter(train_loader))["t"], torch.arange(2))


def test_get_val_loader(config, mock_temporal_data):
    val_loader = config.get_validation_data()
    assert torch.equal(next(iter(val_loader))["t"], torch.arange(2))


def test_get_test_loader(config, mock_temporal_data):
    test_loader = config.get_test_data()
    assert torch.equal(next(iter(test_loader))["t"], torch.arange(2))
