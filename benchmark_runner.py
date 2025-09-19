"""Benchmark runner modes for XAX MNIST mixed precision testing."""

from mnist_mp import MnistClassification, Config
from benchmark_config import BenchmarkConfig
from policy_comparison import PrecisionComparator
from performance_monitor import PerformanceMonitor


def select_precision_policy() -> str:
    """Interactive precision policy selection."""
    policies = BenchmarkConfig.get_all_policies()

    print("\nAvailable precision policies:")
    for i, policy in enumerate(policies, 1):
        description = BenchmarkConfig().get_policy_description(policy)
        print(f"{i}. {policy} - {description}")

    while True:
        try:
            choice = input(f"\nSelect policy (1-{len(policies)}, default=1): ").strip()
            if not choice:
                return policies[0]

            idx = int(choice) - 1
            if 0 <= idx < len(policies):
                return policies[idx]
            else:
                print(f"Please enter a number between 1 and {len(policies)}")
        except ValueError:
            print("Please enter a valid number")


def get_custom_steps() -> int:
    """Get custom number of training steps."""
    while True:
        try:
            steps = input("Enter number of training steps (default=2000): ").strip()
            if not steps:
                return 2000

            steps_int = int(steps)
            if steps_int > 0:
                return steps_int
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")


def run_single_policy_mode():
    """Run single precision policy in interactive mode."""
    print("\n--- Single Policy Mode ---")

    # Select precision policy
    policy = select_precision_policy()

    # Get custom steps
    max_steps = get_custom_steps()

    print(f"\nRunning MNIST training with:")
    print(f"  Precision policy: {policy}")
    print(f"  Max steps: {max_steps}")
    print(f"  Description: {BenchmarkConfig().get_policy_description(policy)}")

    # Create and run config
    config = Config(
        precision_policy=policy,
        max_steps=max_steps,
    )

    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    print(f"\nStarting training...")
    MnistClassification.launch(config)

    end_snapshot = monitor.get_snapshot()
    print("\n" + monitor.format_performance_summary(
        end_snapshot,
        num_epochs=1,  # Epochs aren't tracked, so we can use a placeholder
        num_batches=max_steps
    ))


def select_two_policies():
    """Interactive selection of two precision policies for comparison."""
    policies = BenchmarkConfig.get_all_policies()

    print("\nAvailable precision policies:")
    for i, policy in enumerate(policies, 1):
        description = BenchmarkConfig().get_policy_description(policy)
        print(f"{i}. {policy} - {description}")

    selected_policies = []

    # Select first policy
    while True:
        try:
            choice = input(f"\nSelect FIRST policy (1-{len(policies)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(policies):
                selected_policies.append(policies[idx])
                break
            else:
                print(f"Please enter a number between 1 and {len(policies)}")
        except ValueError:
            print("Please enter a valid number")

    # Select second policy
    while True:
        try:
            choice = input(f"Select SECOND policy (1-{len(policies)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(policies):
                if policies[idx] == selected_policies[0]:
                    print("Please select a different policy from the first one")
                    continue
                selected_policies.append(policies[idx])
                break
            else:
                print(f"Please enter a number between 1 and {len(policies)}")
        except ValueError:
            print("Please enter a valid number")

    return selected_policies


def run_two_policy_mode():
    """Run two policy comparison mode with user-selected policies."""
    print("\n--- Two Policy Comparison Mode ---")
    print("Select two precision policies to compare")

    # Get user's policy selection
    selected_policies = select_two_policies()

    # Get custom steps
    steps = get_custom_steps()

    print(f"\nComparing policies:")
    print(f"  1. {selected_policies[0]} - {BenchmarkConfig().get_policy_description(selected_policies[0])}")
    print(f"  2. {selected_policies[1]} - {BenchmarkConfig().get_policy_description(selected_policies[1])}")
    print(f"  Steps per policy: {steps}")

    # Create custom benchmark config with selected policies
    custom_config = BenchmarkConfig(
        precision_policies=selected_policies,
        benchmark_max_steps=steps,
        print_detailed_results=True
    )

    # Run the comparison
    comparator = PrecisionComparator(custom_config)
    results = comparator.run_comparison()
    comparator.print_comparison_results()

    return results



def run_all_policies_mode():
    """Run and compare all policies."""
    print("\n--- All Policies Comparison Mode ---")

    selected_policies = BenchmarkConfig.get_all_policies()

    # Get custom steps
    steps = get_custom_steps()

    print(f"\nComparing {len(selected_policies)} policies:")
    for i, policy in enumerate(selected_policies, 1):
        description = BenchmarkConfig().get_policy_description(policy)
        print(f"  {i}. {policy} - {description}")

    print(f"  Steps per policy: {steps}")

    # Create config using the classmethod
    custom_config = BenchmarkConfig.create_all_policies_config(max_steps=steps)

    # Run the comparison
    comparator = PrecisionComparator(custom_config)
    results = comparator.run_comparison()
    comparator.print_comparison_results()

    return results