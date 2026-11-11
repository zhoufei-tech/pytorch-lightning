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
from types import ModuleType
from typing import Optional, Union

import torch
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry

# Device types that have a dedicated Lightning accelerator and must not be claimed by the generic one.
_DEDICATED_DEVICE_TYPES = ("cpu", "cuda", "mps", "xla", "meta")


def _current_accelerator_type() -> Optional[str]:
    """Device type of the accelerator PyTorch reports as current, without creating a device context.

    ``check_available=False`` only reads which backend is compiled in or registered (e.g. ``PrivateUse1``), so it is
    safe to call in every rank before the process group is initialized and before devices are pinned.

    """
    if not hasattr(torch, "accelerator"):
        return None
    try:
        device = torch.accelerator.current_accelerator(check_available=False)
    except (RuntimeError, TypeError):
        return None
    return None if device is None else device.type


def _device_module(device_type: str) -> Optional[ModuleType]:
    try:
        return torch.get_device_module(device_type)
    except (RuntimeError, AttributeError):
        return None


class TorchAccelerator(Accelerator):
    """Accelerator for any device backend that integrates with PyTorch's device-generic APIs.

    This covers out-of-tree backends that register themselves with PyTorch (``PrivateUse1`` backends such as
    ``torch_npu`` (``npu``), ``torch_neuronx`` (``neuron``), or ``torch_tpu`` (``tpu``)) and expose:

    - ``torch.accelerator`` (device count, current device index),
    - a device module (``torch.get_device_module(device_type)``) with the usual ``set_device``, ``stream``,
      ``default_stream``, ``memory_stats`` and ``empty_cache`` entry points, and
    - a ``torch.distributed`` backend registered for the device type (picked up via
      ``torch.distributed.Backend.default_device_backend_map``).

    The device type is detected from ``torch.accelerator.current_accelerator()``. Subclasses can pin it by setting
    :attr:`device_type`.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    device_type: Optional[str] = None

    @classmethod
    def _resolve_device_type(cls) -> Optional[str]:
        if cls.device_type is not None:
            return cls.device_type
        device_type = _current_accelerator_type()
        if device_type is None or device_type in _DEDICATED_DEVICE_TYPES:
            return None
        return device_type

    @classmethod
    def _require_device_type(cls) -> str:
        device_type = cls._resolve_device_type()
        if device_type is None:
            raise RuntimeError(
                f"`{cls.__name__}` could not detect a PyTorch accelerator backend. Make sure the backend package for"
                " your device is installed and imported."
            )
        return device_type

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device does not match the accelerator's device type.
        """
        device_type = self._require_device_type()
        if device.type != device_type:
            raise ValueError(f"Device should be {device_type}, got {device} instead.")
        if device.index is not None:
            module = _device_module(device_type)
            if module is not None and hasattr(module, "set_device"):
                module.set_device(device)
            else:
                torch.accelerator.set_device_index(device.index)

    @override
    def teardown(self) -> None:
        device_type = self._resolve_device_type()
        module = _device_module(device_type) if device_type is not None else None
        if module is not None and hasattr(module, "empty_cache"):
            module.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> list[int]:
        """Accelerator device parsing logic.

        ``-1`` or ``"-1"`` means use all devices, an int ``N`` means the first ``N`` devices, and a list or a
        comma-separated string selects specific device indices.

        """
        if isinstance(devices, str):
            devices = devices.strip()
            if devices == "-1":
                devices = -1
            elif "," in devices:
                return [int(x.strip()) for x in devices.split(",") if x.strip()]
            else:
                devices = int(devices)
        if isinstance(devices, int):
            if devices == -1:
                return list(range(torch.accelerator.device_count()))
            return list(range(devices))
        return list(devices)

    @classmethod
    @override
    def get_parallel_devices(cls, devices: Union[int, str, list[int]]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        device_type = cls._require_device_type()
        return [torch.device(device_type, i) for i in cls.parse_devices(devices)]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return torch.accelerator.device_count()

    @classmethod
    @override
    def is_available(cls) -> bool:
        """Whether a supported backend is registered with PyTorch.

        This deliberately does not touch the device: Lightning calls it in every process before the process group is
        initialized, and some runtimes claim all visible devices on first use.

        """
        return cls._resolve_device_type() is not None

    @staticmethod
    @override
    def name() -> str:
        return "torch_accelerator"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )
