"""Policy comparison functionality for mixed precision benchmarking."""

import time
import threading
from typing import List, Optional
from mnist_mp import MnistClassification, Config
from benchmark_config import BenchmarkConfig
from performance_monitor import PerformanceMonitor, BenchmarkResult, compare_benchmark_results


class PrecisionComparator:
    """Orchestrates comparison across multiple precision policies."""

    def __init__(self, benchmark_config: BenchmarkConfig):
        self.benchmark_config = benchmark_config
        self.results: List[BenchmarkResult] = []


    def run_single_policy(self, precision_policy: str) -> Optional[BenchmarkResult]:
        """Run training with a single precision policy and return results."""
        print(f"\n{'='*60}")
        print(f"Testing precision policy: {precision_policy}")
        print(f"Description: {self.benchmark_config.get_policy_description(precision_policy)}")
        print(f"{'='*60}")

        # Create config with the specified precision policy
        config = Config(
            precision_policy=precision_policy,
            max_steps=self.benchmark_config.benchmark_max_steps,
            batch_size=self.benchmark_config.benchmark_batch_size,
        )

        # Initialize performance monitor
        monitor = PerformanceMonitor(
            enable_memory_tracking=self.benchmark_config.enable_memory_tracking,
            enable_timing_tracking=self.benchmark_config.enable_timing_tracking
        )
        monitor.start_monitoring()
        start_time = time.time()

        # Start periodic memory sampling
        stop_sampling = threading.Event()
        sampling_thread = threading.Thread(target=self._sample_memory_periodically,
                                          args=(monitor, stop_sampling, 1.0))
        sampling_thread.daemon = True
        sampling_thread.start()

        try:
            # Run training
            MnistClassification.launch(config, use_cli=False)

        except Exception as e:
            print(f"Error during training with {precision_policy}: {str(e)}")
            return None
        finally:
            # Stop memory sampling
            stop_sampling.set()
            sampling_thread.join(timeout=2.0)

        # Get final performance snapshot
        end_time = time.time()
        end_snapshot = monitor.get_snapshot()
        training_time = end_time - start_time

        # Calculate performance metrics
        delta = monitor.get_delta(end_snapshot)
        throughput = monitor.calculate_throughput(
            self.benchmark_config.benchmark_max_steps,
            delta.get('clock_time_delta', training_time)
        )

        # Get peak memory stats
        peak_stats = monitor.get_peak_memory_stats()

        # Collect and organize benchmark results
        result = BenchmarkResult(
            precision_policy=precision_policy,
            training_time_sec=training_time,
            throughput_batches_per_sec=throughput,
            cpu_memory_mb=end_snapshot.cpu_memory_mb,
            gpu_memory_mb=end_snapshot.gpu_memory_mb,
            cpu_memory_delta_mb=delta.get('cpu_memory_delta_mb', 0.0),
            gpu_memory_delta_mb=delta.get('gpu_memory_delta_mb', 0.0),
            peak_gpu_memory_mb=peak_stats['peak_gpu_mb']
        )


        return result

    def _sample_memory_periodically(self, monitor: PerformanceMonitor, stop_event: threading.Event, interval: float):
        """Periodically sample memory usage during training."""
        while not stop_event.is_set():
            monitor.sample_memory()
            stop_event.wait(interval)

    def run_comparison(self) -> List[BenchmarkResult]:
        """Run comparison between two different precision policies."""
        print(f"Starting precision policy comparison...")
        print(f"Testing {len(self.benchmark_config.precision_policies)} policies")
        print(f"Max steps per policy: {self.benchmark_config.benchmark_max_steps}")
        print(f"Batch size: {self.benchmark_config.benchmark_batch_size}")

        self.results = []

        for i, policy in enumerate(self.benchmark_config.precision_policies, 1):
            print(f"\n[{i}/{len(self.benchmark_config.precision_policies)}] Running {policy}...")

            try:
                result = self.run_single_policy(policy)
                if result is not None:
                    self.results.append(result)
                    print(f"✓ {policy} completed: {result.format_summary()}")
                else:
                    print(f"✗ {policy} failed: Training returned no result")

            except Exception as e:
                print(f"✗ {policy} failed: {str(e)}")
                continue

        return self.results

    def print_comparison_results(self):
        """Print formatted comparison results."""
        if not self.results:
            print("No results to compare.")
            return

        print(f"\n{compare_benchmark_results(self.results)}")

