from statistics import mean
from time import perf_counter

import numpy as np

from employee_manager.analytics import (
    calculate_projections_numpy,
    calculate_projections_sequential,
    generate_employees,
)
from employee_manager.parallel import (
    calculate_projections_processes,
    calculate_projections_threads,
)

SIZES = (5_000, 20_000, 50_000)
REPEATS = 3


def measure(function, *args) -> tuple[list[float], object]:
    times = []
    result = None
    for _ in range(REPEATS):
        start = perf_counter()
        result = function(*args)
        times.append(perf_counter() - start)
    return times, result


def main() -> None:
    print(f"{'dataset':>8} {'method':12} {'workers':>7} {'run 1':>8} {'run 2':>8} {'run 3':>8} {'mean':>8} {'speedup':>8}")
    for size in SIZES:
        salaries = [employee.salary for employee in generate_employees(size)]
        methods = [
            ("Sequential", "1", calculate_projections_sequential, (salaries,)),
            ("Threads", "4", calculate_projections_threads, (salaries, 4)),
            ("Processes", "4", calculate_projections_processes, (salaries, 4)),
            ("NumPy", "-", calculate_projections_numpy, (salaries,)),
        ]
        baseline = None
        reference = None
        for name, workers, function, args in methods:
            times, result = measure(function, *args)
            average = mean(times)
            if baseline is None:
                baseline, reference = average, result
            assert np.allclose(result, reference, rtol=1e-9)
            runs = " ".join(f"{t:8.3f}" for t in times)
            print(f"{size:>8} {name:12} {workers:>7} {runs} {average:8.3f} {baseline / average:7.2f}x")
        print()


if __name__ == "__main__":
    main()