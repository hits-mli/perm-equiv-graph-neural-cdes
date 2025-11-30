import pytest
import numpy as np
from dataset.ode_dataset import ODEDataset


class Config:
    def __init__(self):
        self.name = "heat"
        self.num_nodes = 4
        self.split_ratio = [0.8, 0.2]
        self.final_time = 5.0
        self.time_tick = 100
        self.graph_type = "grid"
        self.operator_type = "norm_lap"
        self.sampling_type = "irregular"
        self.layout = "grid"
        self.dt0 = 0.1  # Overwrites previous 0.01
        self.dynamic_graph = True
        self.all_dynamic = True
        self.seed = 1234
        self.method = "Dopri5"
        self.batch_size = 2
        self.amp_range = (0.5, 1.0)


@pytest.fixture
def basic_config():
    return Config()


def test_ode_dataset_initialization(basic_config):
    dataset = ODEDataset(basic_config)
    assert dataset.sampling_type == "irregular"
    assert dataset.time_tick == 100
    assert dataset.T == 5.0
    assert dataset.batch_size == 2


def test_gen_sampling_time_equal(basic_config):
    dataset = ODEDataset(basic_config)
    times = dataset.gen_sampling_time()
    assert times.shape == (dataset.batch_size, 120)
    assert np.allclose(times[0, 0], 0.0)


def test_gen_sampling_time_irregular(basic_config):
    dataset = ODEDataset(basic_config)
    times = dataset.gen_sampling_time()
    expected_points = int(dataset.time_tick * 1.2)
    assert times.shape == (dataset.batch_size, expected_points)
    assert np.allclose(times[:, 0], 0.0)


def test_split_train_val_test(basic_config):
    dataset = ODEDataset(basic_config)
    id_train, id_test_extra, id_test_inter = dataset.split_train_val_test()
    assert len(id_train) == int(dataset.time_tick * 0.8)
    assert len(id_test_extra) == dataset.time_tick - len(id_train)
    assert id_test_inter is not None


def test_invalid_sampling_type(basic_config):
    basic_config.sampling_type = "invalid"
    with pytest.raises(ValueError):
        dataset = ODEDataset(basic_config)
        dataset.gen_sampling_time()
