"""Main entry point for XAX MNIST mixed precision benchmarking."""

import sys
from benchmark_runner import run_single_policy_mode, run_two_policy_mode, run_all_policies_mode


def print_welcome():
    """Print welcome message and available options."""
    print("=" * 60)
    print("XAX MNIST Mixed Precision Benchmark Suite")
    print("=" * 60)
    print("Available modes:")
    print("1. Single run - Test one precision policy interactively")
    print("2. Two policy comparison - Select and compare 2 policies")
    print("3. All policies comparison - Compare all 7 precision policies")
    print("=" * 60)


def main():
    """Main function with interactive mode selection."""
    print_welcome()

    while True:
        try:
            mode = input("\nSelect mode (1-3, q to quit): ").strip().lower()

            if mode in ['q', 'quit', 'exit']:
                print("Goodbye!")
                sys.exit(0)
            elif mode == '1':
                run_single_policy_mode()
                break
            elif mode == '2':
                run_two_policy_mode()
                break
            elif mode == '3':
                run_all_policies_mode()
                break
            else:
                print("Please enter 1, 2, 3, or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again")


if __name__ == "__main__":
    main()