from __future__ import annotations
import threading
import psutil

import shutil
import os
import torch

import platform

import typing


def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=True):
    # Join all daemon threads, i.e. atexit.register(lambda: join_threads())
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f'Joining thread {t.name}')
            t.join()


def _disk_usage():
    total, used, free = shutil.disk_usage("/")
    return total, used, free


def _total_ram():
    v_mem = psutil.virtual_memory()

    return v_mem.total, v_mem.used


class _MachineResources(object):
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)

    def __init__(self):
        self._default_gpu_device = None

    @property
    def cpu_usage_percentage(self) -> float:
        return psutil.cpu_percent()

    @property
    def readable_info(self) -> str:
        info = self.info
        total_disk = info["disk"]["total"]
        free_disk = info["disk"]["free"]
        used_mem = info["ram"]["used"]
        total_mem = info["ram"]["total"]
        gpu_mem_used = info["gpu_mem"]["used"]
        gpu_mem_total = info["gpu_mem"]["total"]

        s = f"disk {(total_disk - free_disk) / self.gb:.1f}/{total_disk / self.gb:.1f} GB"
        s += f"; ram {used_mem / self.mb:.1f}/{total_mem / self.mb:.1f} MB"
        s += f"; gpu-mem {gpu_mem_used / self.mb:.1f}/{gpu_mem_total / self.mb:.1f} MB"
        return s

    @property
    def info(self) -> dict:
        total_disk, used_disk, free_disk = self.disk_usage
        gpu_mem_used, gpu_mem_total = self.gpu_mem_get_info
        total_mem, used_mem = self.ram

        return {
            "disk": {
                "total": total_disk,
                "used": used_disk,
                "free": free_disk
            },
            "ram": {
                "total": total_mem,
                "used": used_mem
            },
            "gpu_mem": {
                "total": gpu_mem_total,
                "used": gpu_mem_used
            }
        }

    def __str__(self):
        total_disk, used_disk, free_disk = self.disk_usage
        gpu_mem_used, gpu_mem_total = self.gpu_mem_get_info
        total_mem, used_mem = self.ram
        s = f"RAM\t\t: {used_mem / self.gb:.1f}/{total_mem / self.gb:.1f} GB\n"
        s += f"CPU\t\t: {self.cpu_count} cores\n"
        s += f"Disk\t\t: {(total_disk - free_disk) / self.gb:.1f}/{total_disk / self.gb:.1f} GB\n"
        s += f"GPU\t\t: {self.gpu_count} \n"
        s += f"CUDA\t\t: {self.is_cuda_available} \n"
        s += f"GPU Memory\t: {gpu_mem_used / self.gb:.1f}/{gpu_mem_total / self.gb:.1f} GB;"
        s += f" Device -> {torch.cuda.get_device_name(self.default_gpu)}"
        return s

    @property
    def current_thread(self):
        return threading.current_thread()

    @property
    def is_main_thread(self) -> bool:
        return self.current_thread == threading.main_thread()

    @property
    def ram(self) -> typing.Tuple[int, int]:
        return _total_ram()

    @property
    def disk_usage(self) -> typing.Tuple[int, int, int]:
        return _disk_usage()

    @property
    def cpu_count(self) -> int:
        return os.cpu_count()

    @property
    def gpu_count(self) -> int:
        return torch.cuda.device_count()

    @property
    def is_cuda_available(self) -> bool:
        return torch.cuda.is_available()

    def empty_gpu_cache(self) -> None:
        torch.cuda.empty_cache()

    @property
    def list_gpu_processes(self):
        return torch.cuda.list_gpu_processes(self.default_gpu)

    @property
    def gpu_mem_get_info(self):
        return torch.cuda.mem_get_info(self.default_gpu)

    @property
    def default_gpu(self) -> torch.device:
        if self._default_gpu_device is None:
            self._default_gpu_device = self.select_gpu_device(verbose=True)
        return self._default_gpu_device

    @default_gpu.setter
    def default_gpu(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise TypeError(f"Invalid type for a device -> {type(device)}")

        self._default_gpu_device = device

    def select_gpu_device(self, device='', batch_size=0, newline=True, verbose=True) -> torch.device:
        # source -->
        # https://github.com/ultralytics/yolov5/blob/2370a5513ebf67bd10b8d15fd6353e008380bc43/utils/torch_utils.py#L108

        # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
        s = f'Python-{platform.python_version()} torch-{torch.__version__} '
        device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
        cpu = device == 'cpu'
        mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
        if cpu or mps:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
        elif device:  # non-cpu device requested
            os.environ[
                'CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
            assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
                f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

        if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
            devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
            n = len(devices)  # device count
            if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
                assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
            space = ' ' * (len(s) + 1)
            for i, d in enumerate(devices):
                p = torch.cuda.get_device_properties(i)
                s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
            arg = 'cuda:0'
        elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
            s += 'MPS\n'
            arg = 'mps'
        else:  # revert to CPU
            s += 'CPU\n'
            arg = 'cpu'

        if not newline:
            s = s.rstrip()
        if verbose:
            print(s)
        return torch.device(arg)


machine = _MachineResources()

__all__ = ["threaded", "join_threads", "machine"]
