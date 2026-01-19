#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Memory management tool for CPU, GPU, and Disk tensors.
"""

from typing import Any
import gc, os, tempfile
import torch


def clear_mem():
    """清理 CPU 和 GPU 内存"""
    gc.collect()
    torch.cuda.empty_cache()


def show_memory():
    """打印当前 GPU 占用"""
    used_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"Used GPU Memory: {used_memory:.2f} GB")


class Memory:
    """管理单个 tensor，可以在 CPU/GPU/Disk 之间切换"""

    def __init__(self, name: str, value: torch.Tensor, desc: str, path: str) -> None:
        self.name = name
        self.desc = desc
        self.path = path

        if desc == 'cpu':
            self._value = value.cpu()
        elif desc.startswith('cuda'):
            self._value = value.to(desc)
        elif desc == 'disk':
            torch.save(value.cpu(), self.path)
            self._value = None
        else:
            raise ValueError(f"Unsupported desc: {desc}")

        clear_mem()

    def to(self, desc: str = 'cpu') -> "Memory":
        """将 tensor 移动到指定设备或保存到磁盘"""
        if desc == self.desc:
            return self

        if self.desc == 'disk' and desc != 'disk':
            self._value = torch.load(self.path)

        if desc.startswith('cuda'):
            self._value = self._value.to(desc)
        elif desc == 'cpu':
            self._value = self._value.cpu()
        elif desc == 'disk':
            torch.save(self._value.cpu(), self.path)
            del self._value
            self._value = None
        else:
            raise ValueError(f"Cannot move to {desc}")

        self.desc = desc
        clear_mem()
        return self

    def __call__(self) -> torch.Tensor:
        if self.desc == 'disk':
            self._value = torch.load(self.path, map_location='cpu')
            self.desc = 'cpu'
        return self._value

    @property
    def value(self) -> torch.Tensor:
        return self()

    @value.setter
    def value(self, new_value: torch.Tensor):
        assert isinstance(new_value, torch.Tensor), 'Only torch.Tensor is supported'
        if self.desc.startswith('cuda'):
            self._value = new_value.to(self.desc)
        elif self.desc == 'cpu':
            self._value = new_value.cpu()
        elif self.desc == 'disk':
            torch.save(new_value.cpu(), self.path)
        else:
            raise ValueError(f"Cannot set value to {self.desc}")

    def __del__(self):
        try:
            if hasattr(self, "_value"):
                del self._value
        finally:
            clear_mem()


class MemoryBank:
    """管理多个 Memory 对象"""

    def __init__(self, disk_path: str = None) -> None:
        self.record = {}
        self._temp_dir = None
        if disk_path is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            disk_path = self._temp_dir.name
        self.disk_path = disk_path

    def add_value(self, key: str, value: torch.Tensor, desc: str = 'cpu') -> Memory:
        mem = Memory(key, value, desc, os.path.join(self.disk_path, key + '.pth'))
        self.record[key] = mem
        return mem

    def del_value(self, key: str):
        if key in self.record:
            del self.record[key]
        clear_mem()

    def __del__(self):
        for key in list(self.record.keys()):
            self.del_value(key)
        self.record.clear()
        if self._temp_dir:
            self._temp_dir.cleanup()
        clear_mem()


# 全局内存管理
MEMORY_BANK = MemoryBank()

