import time
import math
import sys
from typing import List
import numpy as np
from confidence_calibration import ConfidenceCalibrator

def original_temperature_scale(probs: List[float], temp: float = 1.5) -> List[float]:
    eps = 1e-10
    log_probs = [math.log(max(p, eps)) for p in probs]
    scaled = [lp / temp for lp in log_probs]
    max_s = max(scaled)
    exp_scaled = [math.exp(s - max_s) for s in scaled]
    total = sum(exp_scaled)
    return [e / total for e in exp_scaled]

def run_benchmark():
    sizes = [3, 10, 100, 1000]
    num_iters = 10000
    calibrator = ConfidenceCalibrator()

    print("Benchmark Comparison (Original vs Optimized):\n")
    for size in sizes:
        probs = [1.0/size] * size
        probs[0] = 0.5
        total = sum(probs)
        probs = [p/total for p in probs]

        start = time.perf_counter()
        for _ in range(num_iters):
            original_temperature_scale(probs)
        end = time.perf_counter()
        elapsed_orig = end - start

        start = time.perf_counter()
        for _ in range(num_iters):
            calibrator._temperature_scale(probs)
        end = time.perf_counter()
        elapsed_opt = end - start

        print(f"Array Size {size}:")
        print(f"  Original:  {elapsed_orig:.4f}s")
        print(f"  Optimized: {elapsed_opt:.4f}s")
        print(f"  Speedup:   {elapsed_orig/elapsed_opt:.2f}x\n")
    print("Benchmark complete.")

if __name__ == '__main__':
    run_benchmark()
