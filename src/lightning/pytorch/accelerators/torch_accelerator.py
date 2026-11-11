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
from typing import Any

import torch
from typing_extensions import override

from lightning.fabric.accelerators.torch_accelerator import TorchAccelerator as _TorchAccelerator
from lightning.fabric.accelerators.torch_accelerator import _device_module
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator


class TorchAccelerator(Accelerator, _TorchAccelerator):
    """Accelerator for any device backend that integrates with PyTorch's device-generic APIs.

    See :class:`lightning.fabric.accelerators.torch_accelerator.TorchAccelerator`.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Return the memory stats reported by the backend's device module, if it provides them."""
        device_type = self._require_device_type()
        module = _device_module(device_type)
        if module is None or not hasattr(module, "memory_stats"):
            return {}
        return module.memory_stats(torch.device(device) if isinstance(device, str) else device)
