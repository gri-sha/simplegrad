import pytest
from simplegrad.core.devices import cuda_is_available, default_device

CUDA_AVAILABLE = cuda_is_available()
_DEVICES = ["cpu"] + (["cuda:0"] if CUDA_AVAILABLE else [])


@pytest.fixture(params=_DEVICES)
def device(request):
    default_device(request.param)
    yield request.param
    default_device("cpu")
