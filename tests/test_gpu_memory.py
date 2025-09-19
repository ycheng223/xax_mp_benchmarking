"""Test script to verify nvidia-smi GPU memory measurement."""

import subprocess
import time


def test_nvidia_smi_direct():
    """Test nvidia-smi command directly."""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            memory_mb = float(result.stdout.strip())
            print(f"✓ nvidia-smi GPU memory: {memory_mb:.1f} MB")
            return memory_mb
        else:
            print(f"✗ nvidia-smi failed with return code: {result.returncode}")
            print(f"stderr: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ nvidia-smi error: {e}")
        return None


def test_performance_monitor():
    """Test the updated PerformanceMonitor."""
    try:
        from performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(enable_memory_tracking=True)
        gpu_memory = monitor.get_gpu_memory_mb()

        print(f"✓ PerformanceMonitor GPU memory: {gpu_memory:.1f} MB")
        return gpu_memory
    except Exception as e:
        print(f"✗ PerformanceMonitor error: {e}")
        return None


def compare_with_nvidia_smi_total():
    """Compare with nvidia-smi full output."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n--- nvidia-smi full output (first few lines) ---")
            lines = result.stdout.split('\n')
            for line in lines[:15]:  # Show first 15 lines
                if 'MiB' in line or 'GPU' in line or '|' in line:
                    print(line)
        else:
            print("✗ Could not get nvidia-smi full output")
    except Exception as e:
        print(f"✗ nvidia-smi full output error: {e}")


def main():
    """Test GPU memory measurement improvements."""
    print("Testing GPU Memory Measurement with nvidia-smi")
    print("=" * 50)

    # Test direct nvidia-smi
    print("\n1. Direct nvidia-smi test:")
    nvidia_memory = test_nvidia_smi_direct()

    # Test updated PerformanceMonitor
    print("\n2. PerformanceMonitor test:")
    monitor_memory = test_performance_monitor()

    # Compare results
    if nvidia_memory is not None and monitor_memory is not None:
        diff = abs(nvidia_memory - monitor_memory)
        print(f"\n3. Comparison:")
        print(f"   nvidia-smi:        {nvidia_memory:.1f} MB")
        print(f"   PerformanceMonitor: {monitor_memory:.1f} MB")
        print(f"   Difference:        {diff:.1f} MB")

        if diff < 10:  # Within 10MB is reasonable
            print("   ✓ Values are very close!")
        elif diff < 100:
            print("   ⚠ Values are somewhat different")
        else:
            print("   ✗ Values are significantly different")

    # Show nvidia-smi context
    print("\n4. nvidia-smi context:")
    compare_with_nvidia_smi_total()

    print(f"\n{'='*50}")
    print("GPU memory measurement should now match nvidia-smi more closely!")


if __name__ == "__main__":
    main()