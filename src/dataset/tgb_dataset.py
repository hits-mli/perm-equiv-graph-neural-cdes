import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data.temporal import TemporalData


class SlidingWindowTemporalDataset(Dataset):
    def __init__(self, data: TemporalData, window_size: int, stride: int = 1):
        """
        Args:
            data (TemporalData): Temporal edge list.
            window_size (int): Number of past time steps to include in each batch.
            stride (int): Step size for moving the window.
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride

        # Get sorted unique timestamps
        self.unique_times = torch.unique(self.data.t)

        # Compute number of possible sliding windows
        self.num_batches = max(0, (len(self.unique_times) - window_size) // stride + 1)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        """
        Returns a subgraph corresponding to the sliding window at index `idx`.
        """
        start = idx * self.stride
        time_window = self.unique_times[
            start : start + self.window_size
        ]  # Select timestamps
        mask = torch.isin(self.data.t, time_window)  # Mask edges in this time range

        # Create the subgraph for this batch
        return TemporalData(
            src=self.data.src[mask], dst=self.data.dst[mask], t=self.data.t[mask]
        )


# Custom collate function to avoid default PyTorch collation
def collate_fn(batch):
    """Ensures TemporalData objects are not stacked but kept as a list."""
    return batch[0]  # The batch is a list of TemporalData objects


class SlidingWindowTemporalLoader(DataLoader):
    def __init__(self, dataset: SlidingWindowTemporalDataset, batch_size: int = 1):
        """
        Custom DataLoader for temporal graphs with sliding window batching.

        Args:
            dataset (SlidingWindowTemporalDataset): The dataset containing temporal subgraphs.
            batch_size (int): Number of subgraphs per batch.
        """
        super().__init__(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
