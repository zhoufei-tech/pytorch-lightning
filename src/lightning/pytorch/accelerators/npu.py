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
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.pytorch.accelerators.torch_accelerator import TorchAccelerator

_TORCH_NPU_AVAILABLE = RequirementCache("torch_npu")


class NPUAccelerator(TorchAccelerator):
    """Accelerator for Ascend NPU devices (``torch_npu``).

    Device management, streams, memory stats and the ``hccl`` process group backend all come from the generic
    :class:`~lightning.pytorch.accelerators.torch_accelerator.TorchAccelerator` via ``torch_npu``'s registration with
    PyTorch.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    device_type = "npu"

    @classmethod
    @override
    def is_available(cls) -> bool:
        if not _TORCH_NPU_AVAILABLE:
            return False
        import torch_npu  # noqa: F401

        return super().is_available()

    @staticmethod
    @override
    def name() -> str:
        return "npu"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )
