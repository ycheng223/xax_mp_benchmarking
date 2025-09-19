"""Configuration for benchmarking and precision policy comparison."""

from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""

    # Precision policies to test (XAX built-in policies)
    precision_policies: List[str] = None

    benchmark_max_steps: int = 2000  # Fewer steps for comparison
    benchmark_batch_size: int = 128

    # Performance monitoring settings
    enable_memory_tracking: bool = True
    enable_timing_tracking: bool = True

    # Output settings
    print_detailed_results: bool = True
    save_results_to_file: bool = False

    def __post_init__(self):
        if self.precision_policies is None:
            # Default set of precision policies to test
            self.precision_policies = [
                "full",         # params=float32,compute=float32,output=float32
                "half_param",   # params=float16,compute=float16,output=float32
                "half_compute", # params=float32,compute=float16,output=float32
            ]

    @classmethod
    def get_all_policies(cls):
        """Get all available precision policies for comprehensive testing."""
        return [
            "full",
            "half_param",
            "half_compute",
            "half_output",
            "half_front",
            "half_back",
            "half_total"
        ]

    @classmethod
    def create_all_policies_config(cls, max_steps: int = 2000):
        """Create a benchmark config for testing all precision policies."""
        return cls(
            precision_policies=cls.get_all_policies(),
            benchmark_max_steps=max_steps,
            print_detailed_results=True
        )

    def get_policy_description(self, policy: str) -> str:
        """Get human-readable description of a precision policy."""
        descriptions = {
            "full": "Full precision (float32 everywhere)",
            "half_param": "Half precision parameters, float32 output",
            "half_compute": "Half precision compute, float32 params/output",
            "half_output": "Half precision output only",
            "half_front": "Half precision params and compute",
            "half_back": "Half precision compute and output",
            "half_total": "Half precision everywhere"
        }
        return descriptions.get(policy, f"Unknown policy: {policy}")


