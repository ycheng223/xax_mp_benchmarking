"""Performance monitoring utilities for benchmarking."""

import time
import os
import subprocess
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False



@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    wall_time: float
    clock_time: float
    cpu_memory_mb: float
    gpu_memory_mb: float


class PerformanceMonitor:
    """Monitor performance metrics during training."""

    def __init__(self, enable_memory_tracking: bool = True, enable_timing_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_timing_tracking = enable_timing_tracking
        self.start_snapshot: Optional[PerformanceSnapshot] = None
        self.peak_gpu_memory_mb: float = 0.0
        self.memory_samples: list[float] = []

    def get_cpu_memory_mb(self) -> float:
        """Get current CPU memory usage in MB."""
        if not self.enable_memory_tracking or not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def get_gpu_memory_mb(self) -> float:
        """Get GPU memory usage in MB using nvidia-smi."""
        if not self.enable_memory_tracking:
            return 0.0

        try:
            # Use nvidia-smi to get current GPU memory usage
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass

        return 0.0

    def sample_memory(self):
        """Sample current memory usage and update peak."""
        if not self.enable_memory_tracking:
            return

        current_gpu_memory = self.get_gpu_memory_mb()
        self.memory_samples.append(current_gpu_memory)
        self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, current_gpu_memory)

    def get_peak_memory_stats(self) -> Dict[str, float]:
        """Get peak memory statistics."""
        if not self.memory_samples:
            return {"peak_gpu_mb": 0.0, "avg_gpu_mb": 0.0, "samples_count": 0.0}

        return {
            "peak_gpu_mb": self.peak_gpu_memory_mb,
            "avg_gpu_mb": sum(self.memory_samples) / len(self.memory_samples),
            "samples_count": float(len(self.memory_samples))
        }

    def get_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        wall_time = time.time() if self.enable_timing_tracking else 0.0
        clock_time = time.perf_counter() if self.enable_timing_tracking else 0.0
        cpu_memory = self.get_cpu_memory_mb()
        gpu_memory = self.get_gpu_memory_mb()

        return PerformanceSnapshot(
            wall_time=wall_time,
            clock_time=clock_time,
            cpu_memory_mb=cpu_memory,
            gpu_memory_mb=gpu_memory
        )

    def start_monitoring(self):
        """Start monitoring by taking initial snapshot."""
        self.start_snapshot = self.get_snapshot()

    def get_delta(self, end_snapshot: PerformanceSnapshot) -> Dict[str, float]:
        """Get performance delta from start to end snapshot."""
        if self.start_snapshot is None:
            return {}

        return {
            'wall_time_delta': end_snapshot.wall_time - self.start_snapshot.wall_time,
            'clock_time_delta': end_snapshot.clock_time - self.start_snapshot.clock_time,
            'cpu_memory_delta_mb': end_snapshot.cpu_memory_mb - self.start_snapshot.cpu_memory_mb,
            'gpu_memory_delta_mb': end_snapshot.gpu_memory_mb - self.start_snapshot.gpu_memory_mb,
        }

    def calculate_throughput(self, num_batches: int, elapsed_time: float) -> float:
        """Calculate throughput in batches per second."""
        if elapsed_time <= 0:
            return 0.0
        return num_batches / elapsed_time

    def format_performance_summary(self, end_snapshot: PerformanceSnapshot,
                                 num_epochs: int, num_batches: int) -> str:
        """Format a human-readable performance summary."""
        if self.start_snapshot is None:
            return "No performance data available"

        delta = self.get_delta(end_snapshot)
        throughput = self.calculate_throughput(num_batches, delta['clock_time_delta'])

        lines = [
            "Performance Summary:",
            f"  Training Time: {delta['wall_time_delta']:.2f}s (wall), {delta['clock_time_delta']:.2f}s (clock)",
            f"  Throughput: {throughput:.2f} batches/sec",
        ]

        if PSUTIL_AVAILABLE and self.enable_memory_tracking:
            lines.append(f"  CPU Memory: {end_snapshot.cpu_memory_mb:.1f} MB (Δ{delta['cpu_memory_delta_mb']:+.1f} MB)")

        if end_snapshot.gpu_memory_mb > 0 and self.enable_memory_tracking:
            lines.append(f"  GPU Memory: {end_snapshot.gpu_memory_mb:.1f} MB (Δ{delta['gpu_memory_delta_mb']:+.1f} MB)")

        return "\n".join(lines)


@dataclass
class BenchmarkResult:
    """Results from a single precision policy benchmark run."""
    precision_policy: str
    training_time_sec: float
    throughput_batches_per_sec: float
    cpu_memory_mb: float
    gpu_memory_mb: float
    cpu_memory_delta_mb: float
    gpu_memory_delta_mb: float
    peak_gpu_memory_mb: float

    def format_summary(self) -> str:
        """Format a one-line summary of the benchmark result."""
        return (f"{self.precision_policy:12} | "
                f"Time: {self.training_time_sec:.1f}s | "
                f"Throughput: {self.throughput_batches_per_sec:.1f} b/s | "
                f"GPU Peak: {self.peak_gpu_memory_mb:.0f}MB")


def compare_benchmark_results(results: list[BenchmarkResult]) -> str:
    """Generate a comparison table of benchmark results."""
    if not results:
        return "No benchmark results to compare"

    lines = [
        "=" * 95,
        "PRECISION POLICY PERFORMANCE COMPARISON",
        "=" * 95,
        f"{'Policy':<12} | {'Time':<8} | {'Throughput':<12} | {'CPU Mem':<8} | {'GPU Mem':<8} | {'GPU Peak':<9} | {'GPU Δ':<8}",
        "-" * 95,
    ]

    # Sort by training time ascending (fastest first)
    sorted_results = sorted(results, key=lambda x: x.training_time_sec)

    for result in sorted_results:
        line = (f"{result.precision_policy:<12} | "
                f"{result.training_time_sec:<8.1f} | "
                f"{result.throughput_batches_per_sec:<12.1f} | "
                f"{result.cpu_memory_mb:<8.0f} | "
                f"{result.gpu_memory_mb:<8.0f} | "
                f"{result.peak_gpu_memory_mb:<9.0f} | "
                f"{result.gpu_memory_delta_mb:<8.0f}")
        lines.append(line)

    # Add peak memory stats
    lowest_peak = min(results, key=lambda x: x.peak_gpu_memory_mb)

    lines.extend([
        "-" * 95,
        f"Fastest training: {sorted_results[0].precision_policy} ({sorted_results[0].training_time_sec:.1f}s)",
        f"Highest throughput: {max(results, key=lambda x: x.throughput_batches_per_sec).precision_policy} "
        f"({max(results, key=lambda x: x.throughput_batches_per_sec).throughput_batches_per_sec:.1f} b/s)",
        f"Lowest peak memory: {lowest_peak.precision_policy} ({lowest_peak.peak_gpu_memory_mb:.0f}MB)",
        "=" * 95,
    ])

    return "\n".join(lines)