import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.runner import jobs, registry

STARTED_AT = time.time()
CPU_SAMPLE_SECONDS = 0.25

# Our key -> nvidia-smi query field.
GPU_FIELDS = {
    "index": "index",
    "name": "name",
    "utilization_pct": "utilization.gpu",
    "memory_utilization_pct": "utilization.memory",
    "memory_used_mb": "memory.used",
    "memory_total_mb": "memory.total",
    "temperature_c": "temperature.gpu",
    "power_w": "power.draw",
    "power_limit_w": "power.limit",
    "fan_pct": "fan.speed",
    "clock_sm_mhz": "clocks.sm",
    "clock_mem_mhz": "clocks.mem",
    "pstate": "pstate",
    "driver_version": "driver_version",
}
PACKAGES = ["fastapi", "uvicorn", "pydantic", "torch", "diffusers", "transformers", "accelerate"]


def _num(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value  # "[N/A]", names, versions


def gpus() -> list[dict[str, Any]] | dict[str, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS.values())}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    return [
        dict(zip(GPU_FIELDS, (_num(v.strip()) for v in line.split(","))))
        for line in out.strip().splitlines()
    ]


def _proc_mb(path: str, **keys: str) -> dict[str, int] | dict[str, str]:
    """Read kB fields from a Linux /proc file as MB, renamed {our_key: proc_key}."""
    try:
        fields = dict(line.split(":", 1) for line in Path(path).read_text().splitlines())
    except OSError as exc:
        return {"error": str(exc)}
    return {ours: int(fields[theirs].split()[0]) // 1024 for ours, theirs in keys.items() if theirs in fields}


def torch_cuda() -> dict[str, Any] | None:
    # Only once a model has imported torch; don't import it just for stats.
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return {
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "allocated_mb": torch.cuda.memory_allocated() // 2**20,
        "reserved_mb": torch.cuda.memory_reserved() // 2**20,
        "max_allocated_mb": torch.cuda.max_memory_allocated() // 2**20,
    }


def queue_stats() -> dict[str, Any]:
    views = jobs.list()
    counts = {s: 0 for s in ("queued", "running", "completed", "failed")}
    for v in views:
        counts[v["status"]] += 1
    run_times = [v["run_seconds"] for v in views if v["status"] == "completed"]
    running = next((v for v in views if v["status"] == "running"), None)
    return {
        **counts,
        "total": len(views),
        "avg_run_seconds": round(sum(run_times) / len(run_times), 1) if run_times else None,
        "current_job": running and {"id": running["id"], "progress": running["progress"]},
    }


def models_stats() -> list[dict[str, Any]]:
    return [
        {"id": m.id, "task": m.task, "enabled": m.enabled, "loaded": registry.is_loaded(m.id)}
        for m in registry.models.values()
    ]


def gpu_stats() -> dict[str, Any]:
    return {"devices": gpus(), "torch": torch_cuda()}


def _cpu_times() -> dict[str, tuple[int, int]]:
    """{"cpu": (idle, total), "cpu0": ..., ...} in jiffies from /proc/stat."""
    times = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu"):
            name, *values = line.split()
            # user nice system idle iowait irq softirq steal (guest is already counted in user)
            v = [int(x) for x in values[:8]]
            times[name] = (v[3] + v[4], sum(v))
    return times


def _busy_pct(before: tuple[int, int], after: tuple[int, int]) -> float:
    idle, total = after[0] - before[0], after[1] - before[1]
    return round(100 * (1 - idle / total), 1) if total else 0.0


def system_stats() -> dict[str, Any]:
    # CPU % needs two readings; sample over a short window.
    try:
        before, wall0, proc0 = _cpu_times(), time.monotonic(), os.times()
        time.sleep(CPU_SAMPLE_SECONDS)
        after, wall1, proc1 = _cpu_times(), time.monotonic(), os.times()
    except OSError as exc:
        cpu, process_cpu_pct = {"error": str(exc)}, None
    else:
        cpu = {
            "count": os.cpu_count(),
            "usage_pct": _busy_pct(before["cpu"], after["cpu"]),
            "per_core_pct": [_busy_pct(before[k], after[k]) for k in after if k != "cpu"],
            "load_avg_1_5_15": os.getloadavg(),
        }
        # Can exceed 100 when using several cores.
        busy = (proc1.user + proc1.system) - (proc0.user + proc0.system)
        process_cpu_pct = round(100 * busy / (wall1 - wall0), 1)

    memory = _proc_mb(
        "/proc/meminfo",
        total_mb="MemTotal", available_mb="MemAvailable", swap_total_mb="SwapTotal", swap_free_mb="SwapFree",
    )
    if "total_mb" in memory:
        memory["used_mb"] = memory["total_mb"] - memory["available_mb"]
        memory["usage_pct"] = round(100 * memory["used_mb"] / memory["total_mb"], 1)
        memory["swap_used_mb"] = memory["swap_total_mb"] - memory["swap_free_mb"]

    return {
        "cpu": cpu,
        "memory": memory,
        "process": {
            "pid": os.getpid(),
            "uptime_seconds": round(time.time() - STARTED_AT),
            "cpu_pct": process_cpu_pct,
            **_proc_mb("/proc/self/status", rss_mb="VmRSS"),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
            "threads": len(os.listdir("/proc/self/task")) if os.path.isdir("/proc/self/task") else None,
        },
    }


def disk_stats() -> dict[str, Any]:
    disk = shutil.disk_usage(settings.outputs_dir)
    outputs = list(settings.outputs_dir.glob("*.mp4"))
    return {
        "path": str(settings.outputs_dir),
        "total_gb": round(disk.total / 2**30, 1),
        "used_gb": round(disk.used / 2**30, 1),
        "free_gb": round(disk.free / 2**30, 1),
        "videos": len(outputs),
        "videos_size_mb": round(sum(p.stat().st_size for p in outputs) / 2**20, 1),
    }


def info_stats() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in PACKAGES},
        "settings": {
            "cpu_offload": settings.cpu_offload,
            "default_image_model": settings.default_image_model,
            "default_video_model": settings.default_video_model,
        },
    }


# One table drives both GET /v1/stats (all) and GET /v1/stats/{section}.
SECTIONS = {
    "queue": queue_stats,
    "models": models_stats,
    "gpu": gpu_stats,
    "system": system_stats,
    "disk": disk_stats,
    "info": info_stats,
}
Section = StrEnum("Section", list(SECTIONS))


def server_stats() -> dict[str, Any]:
    return {name: fn() for name, fn in SECTIONS.items()}


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
