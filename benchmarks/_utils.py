"""Shared benchmark/profiling helpers for M1.

Deliberately dependency-free: the machine metrics come from ``ctypes`` and the
standard library, so the benchmark runs on a bare Python + NumPy install.  Nothing
here is imported by ``game2048.fast_env`` -- the FastEnv core depends only on the
standard library, NumPy and M0.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

#: Fixed benchmark seed required by the M1 specification.
BENCHMARK_SEED = 20260918


# --------------------------------------------------------------------------- #
# Machine metadata
# --------------------------------------------------------------------------- #


def _total_ram_bytes() -> int:
    """Total physical memory in bytes, or 0 when it cannot be read."""
    try:
        if sys.platform == "win32":
            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        else:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size)
    except Exception:  # pragma: no cover - platform specific
        return 0
    return 0


def _physical_cores() -> int:
    try:
        if sys.platform == "win32":
            count = ctypes.windll.kernel32.GetActiveProcessorCount(0xFFFF)
            if count:
                return int(count)
    except Exception:  # pragma: no cover - platform specific
        pass
    return int(os.cpu_count() or 1)


def _cpu_model() -> str:
    name = platform.processor() or platform.machine() or "unknown"
    try:
        if sys.platform == "win32":
            import winreg  # noqa: PLC0415

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if value:
                name = str(value).strip()
    except Exception:  # pragma: no cover - platform specific
        pass
    return name


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # pragma: no cover - git may be unavailable
        pass
    return "unknown"


def environment_metadata() -> Dict[str, object]:
    """Everything the M1 report must record about the measurement machine."""
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "cpu": _cpu_model(),
        "physical_cores": _physical_cores(),
        "logical_cores": int(os.cpu_count() or 1),
        "ram_bytes": _total_ram_bytes(),
        "git_commit": _git_commit(),
        "seed": BENCHMARK_SEED,
    }


def format_metadata(metadata: Dict[str, object]) -> str:
    ram = metadata["ram_bytes"]
    ram_text = f"{ram / (1024 ** 3):.1f} GiB" if isinstance(ram, int) and ram else "unknown"
    return "\n".join(
        [
            f"OS            : {metadata['os']}",
            f"Python        : {metadata['python']} ({metadata['python_implementation']})",
            f"NumPy         : {metadata['numpy']}",
            f"CPU           : {metadata['cpu']}",
            f"physical cores: {metadata['physical_cores']}",
            f"logical cores : {metadata['logical_cores']}",
            f"RAM           : {ram_text}",
            f"git commit    : {metadata['git_commit']}",
            f"seed          : {metadata['seed']}",
        ]
    )


# --------------------------------------------------------------------------- #
# Resource sampling (CPU utilisation + peak RAM)
# --------------------------------------------------------------------------- #


class ResourceSampler:
    """Sample process CPU utilisation and peak RSS on a background thread.

    CPU utilisation is reported as a fraction of a single logical core, so a
    value above 1.0 means more than one core was busy.  ``peak_rss`` is the
    process high-water mark when the platform exposes one, otherwise the largest
    sample seen.
    """

    def __init__(self, interval: float = 0.05) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_samples: List[float] = []
        self.rss_samples: List[int] = []
        self.peak_rss: int = 0

    def __enter__(self) -> "ResourceSampler":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def start(self) -> None:
        self._cpu_start = time.process_time()
        self._wall_start = time.perf_counter()
        self.peak_rss = current_rss_bytes()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.rss_samples.append(current_rss_bytes())
            self.cpu_samples.append(time.process_time())

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cpu_end = time.process_time()
        self._wall_end = time.perf_counter()
        observed = peak_rss_bytes()
        if observed:
            self.peak_rss = max(self.peak_rss, observed)
        if self.rss_samples:
            self.peak_rss = max(self.peak_rss, max(self.rss_samples))

    @property
    def cpu_seconds(self) -> float:
        return self._cpu_end - self._cpu_start

    @property
    def wall_seconds(self) -> float:
        return self._wall_end - self._wall_start

    @property
    def cpu_utilization(self) -> float:
        """CPU seconds per wall second (1.0 == one saturated logical core)."""
        wall = self.wall_seconds
        return self.cpu_seconds / wall if wall > 0 else 0.0


def current_rss_bytes() -> int:
    """Resident set size of this process, or 0 when unavailable."""
    try:
        if sys.platform == "win32":
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_uint32),
                    ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return int(counters.WorkingSetSize)
        else:
            with open("/proc/self/statm", "r", encoding="utf-8") as handle:
                fields = handle.read().split()
            return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # pragma: no cover - platform specific
        return 0
    return 0


def peak_rss_bytes() -> int:
    """Process high-water RSS mark when the platform exposes it, else 0."""
    try:
        if sys.platform == "win32":
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_uint32),
                    ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return int(counters.PeakWorkingSetSize)
        else:
            import resource  # noqa: PLC0415

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return int(usage.ru_maxrss) * 1024
    except Exception:  # pragma: no cover - platform specific
        return 0
    return 0


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


@dataclass
class Timing:
    """Median/min/max of a repeated measurement, plus resource usage."""

    label: str
    median_s: float
    min_s: float
    max_s: float
    wall_s: float
    cpu_utilization: float
    peak_rss_bytes: int
    repeats: int
    warmup: int
    units: float = 1.0
    unit_name: str = "op"

    @property
    def median_rate(self) -> float:
        return self.units / self.median_s if self.median_s > 0 else float("inf")

    @property
    def min_rate(self) -> float:
        return self.units / self.max_s if self.max_s > 0 else float("inf")

    @property
    def max_rate(self) -> float:
        return self.units / self.min_s if self.min_s > 0 else float("inf")


def time_callable(
    label: str,
    body: Callable[[], None],
    *,
    units: float,
    unit_name: str,
    warmup: int = 5,
    repeats: int = 3,
) -> Timing:
    """Warm up, then time ``body`` ``repeats`` times and report median/min/max."""
    for _ in range(warmup):
        body()

    samples: List[float] = []
    with ResourceSampler() as sampler:
        for _ in range(repeats):
            start = time.perf_counter()
            body()
            samples.append(time.perf_counter() - start)

    values = np.asarray(samples, dtype=np.float64)
    return Timing(
        label=label,
        median_s=float(np.median(values)),
        min_s=float(values.min()),
        max_s=float(values.max()),
        wall_s=sampler.wall_seconds,
        cpu_utilization=sampler.cpu_utilization,
        peak_rss_bytes=sampler.peak_rss,
        repeats=repeats,
        warmup=warmup,
        units=units,
        unit_name=unit_name,
    )


def format_rate(value: float) -> str:
    """Human-readable rate with a compact SI-ish suffix."""
    if value >= 1e9:
        return f"{value / 1e9:,.2f} G"
    if value >= 1e6:
        return f"{value / 1e6:,.2f} M"
    if value >= 1e3:
        return f"{value / 1e3:,.2f} k"
    return f"{value:,.1f} "


def format_bytes(value: int) -> str:
    if not value:
        return "unknown"
    if value >= 1024 ** 3:
        return f"{value / (1024 ** 3):.2f} GiB"
    if value >= 1024 ** 2:
        return f"{value / (1024 ** 2):.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} B"


def format_timing(timing: Timing) -> str:
    return (
        f"{timing.label:<26} median {timing.median_s * 1e3:9.3f} ms  "
        f"min {timing.min_s * 1e3:9.3f}  max {timing.max_s * 1e3:9.3f}  "
        f"{format_rate(timing.median_rate)}{timing.unit_name}/s  "
        f"cpu {timing.cpu_utilization:5.2f}x  "
        f"ram {format_bytes(timing.peak_rss_bytes)}"
    )


def rows_to_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    """Render a list of rows as a simple aligned text table."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    lines = [
        "  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        )
    return "\n".join(lines)
