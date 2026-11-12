import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_cpu_case():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20261112)
    yield
    torch.set_num_threads(previous_threads)
