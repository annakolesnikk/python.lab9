import csv
import os
import tempfile
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np

from employee_manager.analytics import (
    PayrollAnalytics,
    calculate_projections_numpy,
    calculate_projections_sequential,
    department_statistics_numpy,
    department_statistics_python,
    generate_employees,
    payroll_statistics_numpy,
    payroll_statistics_python,
)
from employee_manager.parallel import (
    calculate_projections_processes,
    calculate_projections_threads,
    export_by_department,
    load_files_sequential,
    load_files_threads,
    race_condition_demo,
)

DATA_SIZE = 200_000
PROJECTION_SIZE = 20_000
REPEATS = 5
RESULTS = Path(__file__).parent / "results" / "benchmark_results.csv"
rows: list[dict[str, Any]] = []


def benchmark_repeated(function: Callable[..., Any], *args: Any, repeats: int = REPEATS) -> tuple[float, Any]:
    times = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function(*args)
        times.append(perf_counter() - start)
    return mean(times), result


def peak_memory(function: Callable[..., Any], *args: Any) -> float:
    tracemalloc.start()
    function(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def record(
    experiment: str,
    size: Any,
    workers: Any,
    seconds: float,
    baseline: float,
    memory: str = "-",
    show: bool = True,
) -> None:
    speedup = baseline / seconds
    rows.append(
        {
            "experiment": experiment,
            "data_size": size,
            "workers": workers,
            "mean_time_s": round(seconds, 6),
            "peak_memory_mb": memory,
            "speedup": round(speedup, 2),
        }
    )
    if show:
        print(f"{experiment:28} {str(workers):>7} {seconds:12.6f} s {memory:>10} {speedup:8.2f}x")


def header() -> None:
    print(f"{'experiment':28} {'workers':>7} {'mean time':>14} {'peak MB':>10} {'speedup':>9}")


def main() -> None:
    print(f"CPU count: {os.cpu_count()}")
    employees = generate_employees(DATA_SIZE)
    salaries = [employee.salary for employee in employees]
    salaries_array = np.asarray(salaries)
    departments_array = np.asarray([employee.department for employee in employees])

    print(f"\n1. Payroll statistics ({DATA_SIZE} employees, {REPEATS} repeats)")
    header()
    base, python_stats = benchmark_repeated(payroll_statistics_python, salaries)
    record("Python loops", DATA_SIZE, 1, base, base, f"{peak_memory(payroll_statistics_python, salaries):.2f}")
    numpy_time, numpy_stats = benchmark_repeated(payroll_statistics_numpy, salaries_array)
    record("NumPy", DATA_SIZE, "-", numpy_time, base, f"{peak_memory(payroll_statistics_numpy, salaries_array):.2f}")
    assert all(abs(python_stats[key] - numpy_stats[key]) < 1e-6 * max(1, abs(python_stats[key])) for key in python_stats)

    print(f"\n2. Department statistics ({DATA_SIZE} employees)")
    header()
    base, python_departments = benchmark_repeated(department_statistics_python, employees)
    record("Python loops", DATA_SIZE, 1, base, base)
    numpy_time, numpy_departments = benchmark_repeated(department_statistics_numpy, departments_array, salaries_array)
    record("NumPy (bincount)", DATA_SIZE, "-", numpy_time, base)
    assert all(abs(python_departments[d]["total"] - numpy_departments[d]["total"]) < 1e-3 for d in python_departments)

    print(f"\n3. CPU-bound salary projection ({PROJECTION_SIZE} salaries)")
    header()
    sample = salaries[:PROJECTION_SIZE]
    base, sequential = benchmark_repeated(calculate_projections_sequential, sample)
    record("Sequential", PROJECTION_SIZE, 1, base, base)
    threads_time, threads = benchmark_repeated(calculate_projections_threads, sample, 4)
    record("ThreadPoolExecutor", PROJECTION_SIZE, 4, threads_time, base)
    processes_time, processes = benchmark_repeated(calculate_projections_processes, sample, 4)
    record("ProcessPoolExecutor", PROJECTION_SIZE, 4, processes_time, base)
    numpy_time, vectorized = benchmark_repeated(calculate_projections_numpy, sample)
    record("NumPy (vectorized)", PROJECTION_SIZE, "-", numpy_time, base)
    assert threads == sequential and processes == sequential
    assert np.allclose(vectorized, sequential, rtol=1e-9)
    print("Correctness check: all implementations return the same results")

    print("\n4. Workers scaling (CPU-bound, 3 repeats)")
    print(f"{'workers':>7} {'threads, s':>12} {'processes, s':>14}")
    for workers in (1, 2, 4, 8):
        t, _ = benchmark_repeated(calculate_projections_threads, sample, workers, repeats=3)
        p, _ = benchmark_repeated(calculate_projections_processes, sample, workers, repeats=3)
        print(f"{workers:>7} {t:12.4f} {p:14.4f}")
        record(f"Threads x{workers}", PROJECTION_SIZE, workers, t, base, show=False)
        record(f"Processes x{workers}", PROJECTION_SIZE, workers, p, base, show=False)

    print("\n5. I/O-bound parallel import (8 files, 0.2 s simulated storage latency)")
    header()
    with tempfile.TemporaryDirectory() as directory:
        paths = export_by_department(employees, Path(directory))
        base, loaded = benchmark_repeated(load_files_sequential, paths, 0.2, repeats=3)
        record("Sequential import", len(paths), 1, base, base)
        t, loaded_threads = benchmark_repeated(load_files_threads, paths, 8, 0.2, repeats=3)
        record("ThreadPoolExecutor import", len(paths), 8, t, base)
        assert sorted(e.name for e in loaded) == sorted(e.name for e in loaded_threads)

    print("\n6. Caching of department statistics")
    header()
    analytics = PayrollAnalytics(employees)
    start = perf_counter()
    cold = analytics.statistics("IT")
    cold_time = perf_counter() - start
    record("Cold cache (first call)", DATA_SIZE, "-", cold_time, cold_time)
    start = perf_counter()
    analytics.statistics("IT")
    warm_time = perf_counter() - start
    record("Warm cache (repeated call)", DATA_SIZE, "-", warm_time, cold_time)
    print(analytics.department_statistics.cache_info())
    index = next(i for i, e in enumerate(employees) if e.department == "IT")
    analytics.update_salary(index, 1_000_000)
    after = analytics.statistics("IT")
    print(f"Max IT salary before update: {cold['maximum']:.2f}, after update: {after['maximum']:.2f}")
    print(analytics.department_statistics.cache_info())

    print("\n7. Race condition (8 threads x 1000 increments, expected 8000)")
    unsafe, safe = race_condition_demo()
    print(f"Without Lock: {unsafe}    With Lock: {safe}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {RESULTS}")


if __name__ == "__main__":
    main()