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

from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.accelerators import NPUAccelerator, TorchAccelerator
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector

_DEVICE_TYPE = "privateuseone"


@pytest.fixture
def fake_backend(monkeypatch):
    """Pretend an out-of-tree backend is registered, without touching global PyTorch state."""
    module = SimpleNamespace(
        set_device=mock.Mock(),
        empty_cache=mock.Mock(),
        memory_stats=mock.Mock(return_value={"allocated_bytes.all.current": 1}),
        stream=mock.Mock(side_effect=lambda s: nullcontext(s)),
        default_stream=mock.Mock(return_value="default-stream"),
        Stream=mock.Mock(return_value="side-stream"),
    )
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
    monkeypatch.setitem(torch.distributed.Backend.default_device_backend_map, _DEVICE_TYPE, "fakeccl")
    return module


def test_registered_by_name():
    from lightning.pytorch.accelerators import AcceleratorRegistry

    assert AcceleratorRegistry["torch_accelerator"]["accelerator"] is TorchAccelerator
    assert AcceleratorRegistry["npu"]["accelerator"] is NPUAccelerator
    assert issubclass(NPUAccelerator, TorchAccelerator)


def test_device_stats(fake_backend):
    assert TorchAccelerator().get_device_stats(torch.device(_DEVICE_TYPE, 0)) == {"allocated_bytes.all.current": 1}


@pytest.mark.parametrize("accelerator", ["torch_accelerator", TorchAccelerator()])
def test_single_device_root_is_the_accelerator_device(fake_backend, accelerator):
    """`devices=1` on a non-CUDA accelerator must not fall back to a CPU root device."""
    connector = _AcceleratorConnector(accelerator=accelerator, devices=1)
    assert isinstance(connector.strategy, SingleDeviceStrategy)
    assert connector.strategy.root_device == torch.device(_DEVICE_TYPE, 0)


def test_auto_selects_torch_accelerator(fake_backend, monkeypatch):
    monkeypatch.setattr("lightning.fabric.accelerators.cuda.num_cuda_devices", lambda: 0)
    monkeypatch.setattr("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", staticmethod(lambda: False))
    monkeypatch.setattr("lightning.fabric.accelerators.xla.XLAAccelerator.is_available", staticmethod(lambda: False))
    connector = _AcceleratorConnector(accelerator="auto", devices=1)
    assert isinstance(connector.accelerator, TorchAccelerator)


@pytest.mark.parametrize("precision", ["16-mixed", "bf16-mixed"])
def test_amp_uses_accelerator_device_type(fake_backend, precision):
    connector = _AcceleratorConnector(accelerator="torch_accelerator", devices=1, precision=precision)
    assert connector.precision_plugin.device == _DEVICE_TYPE


def test_fsdp_allowed_on_torch_accelerator(fake_backend):
    connector = _AcceleratorConnector(accelerator="torch_accelerator", devices=2, strategy="fsdp")
    assert connector.strategy.root_device.type == _DEVICE_TYPE


@mock.patch("lightning.pytorch.strategies.ddp.DistributedDataParallel")
def test_ddp_uses_torch_backend_map_and_device_stream(ddp_mock, fake_backend):
    strategy = DDPStrategy(
        accelerator=TorchAccelerator(),
        parallel_devices=TorchAccelerator.get_parallel_devices(2),
        cluster_environment=LightningEnvironment(),
    )
    assert strategy._get_process_group_backend() == "fakeccl"
    strategy._setup_model(torch.nn.Linear(2, 2))
    fake_backend.stream.assert_called_once_with("default-stream")
    ddp_mock.assert_called_once()
    assert ddp_mock.call_args.kwargs["device_ids"] == [0]


def test_npu_accelerator_uses_hccl_via_torch_registration(monkeypatch):
    """`torch_npu` registers `npu -> hccl` with `torch.distributed`; no NPU-specific backend logic is needed."""
    from lightning.fabric.utilities.distributed import _get_default_process_group_backend_for_device

    monkeypatch.setitem(torch.distributed.Backend.default_device_backend_map, "privateuseone", "hccl")
    assert _get_default_process_group_backend_for_device(torch.device("privateuseone", 0)) == "hccl"
    assert NPUAccelerator._require_device_type() == "npu"


def test_npu_not_available_without_torch_npu(monkeypatch):
    monkeypatch.setattr("lightning.pytorch.accelerators.npu._TORCH_NPU_AVAILABLE", False)
    assert not NPUAccelerator.is_available()
