# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from lightning.fabric.accelerators.torch_accelerator import TorchAccelerator
from lightning.fabric.utilities.distributed import (
    _ddp_init_stream_context,
    _get_default_process_group_backend_for_device,
)

# `privateuseone` is the built-in device type of PyTorch's out-of-tree backend slot; it can be constructed on any build.
_DEVICE_TYPE = "privateuseone"


def _fake_device_module(capturing=False):
    return SimpleNamespace(
        set_device=mock.Mock(),
        empty_cache=mock.Mock(),
        stream=mock.Mock(side_effect=lambda s: nullcontext(s)),
        default_stream=mock.Mock(return_value="default-stream"),
        Stream=mock.Mock(return_value="side-stream"),
        is_current_stream_capturing=mock.Mock(return_value=capturing),
    )


@pytest.fixture
def fake_backend(monkeypatch):
    """Pretend an out-of-tree backend is registered, without touching global PyTorch state."""
    module = _fake_device_module()
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda check_available=False: torch.device(_DEVICE_TYPE)
    )
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 4)
    real_get_device_module = torch.get_device_module
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device=None: module if str(device) == _DEVICE_TYPE else real_get_device_module(device),
    )
    return module


def test_not_available_without_backend(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda check_available=False: None)
    assert not TorchAccelerator.is_available()
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda check_available=False: torch.device("cpu"))
    assert not TorchAccelerator.is_available()


@pytest.mark.parametrize("device_type", ["cuda", "mps", "xla"])
def test_does_not_claim_dedicated_accelerators(monkeypatch, device_type):
    """CUDA/MPS/XLA keep their dedicated accelerators."""
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda check_available=False: torch.device(device_type)
    )
    assert not TorchAccelerator.is_available()


def test_is_available_does_not_initialize_the_device(fake_backend, monkeypatch):
    """`is_available` runs in every rank before DDP init; it must only read metadata."""
    monkeypatch.setattr(torch.accelerator, "device_count", mock.Mock(side_effect=AssertionError("device touched")))
    assert TorchAccelerator.is_available()


@pytest.mark.parametrize(
    ("devices", "expected"),
    [(1, [0]), (2, [0, 1]), (-1, [0, 1, 2, 3]), ("-1", [0, 1, 2, 3]), ("3", [0, 1, 2]), ("1,3", [1, 3]), ([2], [2])],
)
def test_parse_and_parallel_devices(fake_backend, devices, expected):
    assert TorchAccelerator.parse_devices(devices) == expected
    assert TorchAccelerator.get_parallel_devices(devices) == [torch.device(_DEVICE_TYPE, i) for i in expected]
    assert TorchAccelerator.auto_device_count() == 4


def test_setup_device_and_teardown(fake_backend):
    accelerator = TorchAccelerator()
    accelerator.setup_device(torch.device(_DEVICE_TYPE, 2))
    fake_backend.set_device.assert_called_once_with(torch.device(_DEVICE_TYPE, 2))
    with pytest.raises(ValueError, match="Device should be"):
        accelerator.setup_device(torch.device("cpu"))
    accelerator.teardown()
    fake_backend.empty_cache.assert_called_once()


def test_pinned_device_type(monkeypatch):
    class PinnedAccelerator(TorchAccelerator):
        device_type = "npu"

    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda check_available=False: None)
    assert PinnedAccelerator.is_available()
    assert PinnedAccelerator._require_device_type() == "npu"


def test_default_process_group_backend_uses_torch_registration(monkeypatch):
    monkeypatch.setitem(torch.distributed.Backend.default_device_backend_map, _DEVICE_TYPE, "fakeccl")
    assert _get_default_process_group_backend_for_device(torch.device(_DEVICE_TYPE, 0)) == "fakeccl"
    assert _get_default_process_group_backend_for_device(torch.device("cpu")) == "gloo"


def test_ddp_stream_context_uses_device_module(fake_backend):
    with _ddp_init_stream_context(torch.device(_DEVICE_TYPE, 0)) as stream:
        assert stream == "default-stream"
    fake_backend.Stream.assert_not_called()

    fake_backend.is_current_stream_capturing.return_value = True
    with _ddp_init_stream_context(torch.device(_DEVICE_TYPE, 0)) as stream:
        assert stream == "side-stream"


def test_ddp_stream_context_without_stream_api(monkeypatch):
    """Backends without a stream API (and CPU) get a null context instead of a CUDA call."""
    real_get_device_module = torch.get_device_module
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device=None: SimpleNamespace() if str(device) == _DEVICE_TYPE else real_get_device_module(device),
    )
    assert isinstance(_ddp_init_stream_context(torch.device(_DEVICE_TYPE, 0)), nullcontext)
    assert isinstance(_ddp_init_stream_context(torch.device("cpu")), nullcontext)
