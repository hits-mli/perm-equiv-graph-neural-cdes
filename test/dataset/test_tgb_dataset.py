import pytest
import torch
from torch_geometric.data.temporal import TemporalData
from torch.utils.data import DataLoader
from dataset.tgb_dataset import (
    SlidingWindowTemporalDataset,
    SlidingWindowTemporalLoader,
)


@pytest.fixture
def temporal_data():
    """Creates a sample TemporalData object for testing."""
    return TemporalData(
        src=torch.tensor(list(range(30))),
        dst=torch.tensor(list(range(30))) + 1,
        t=torch.tensor(list(range(30))) + 1,
    )


def test_dataset_length(temporal_data):
    """Check that the dataset length is computed correctly."""
    dataset = SlidingWindowTemporalDataset(temporal_data, window_size=5, stride=1)
    expected_batches = len(torch.unique(temporal_data.t)) - 5 + 1  # num_batches formula
    assert (
        len(dataset) == expected_batches
    ), f"Expected {expected_batches}, got {len(dataset)}"


def test_sliding_window_batches(temporal_data):
    """Check if batches contain the correct timestamps."""
    dataset = SlidingWindowTemporalDataset(temporal_data, window_size=5, stride=1)

    for i in range(len(dataset)):
        batch = dataset[i]
        expected_timestamps = list(range(i + 1, i + 6))
        assert batch.t is not None, f"Batch {i} timestamps are None"
        assert (
            batch.t.tolist() == expected_timestamps
        ), f"Batch {i} timestamps incorrect: {batch.t.tolist()}"


def test_dataloader_integration(temporal_data):
    """Ensure the DataLoader properly loads the dataset without errors."""
    dataset = SlidingWindowTemporalDataset(temporal_data, window_size=5, stride=1)
    loader = SlidingWindowTemporalLoader(dataset, batch_size=1)

    all_batches = []
    for batch in loader:  # Since batch_size=1, it's a list with one TemporalData object
        all_batches.append(batch.t.tolist())

    # Expected timestamps per batch
    expected_batches = [list(range(i + 1, i + 6)) for i in range(len(dataset))]

    assert (
        all_batches == expected_batches
    ), f"DataLoader produced incorrect batches: {all_batches}"


if __name__ == "__main__":
    pytest.main()
