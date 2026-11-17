# Employee Manager

Laboratory work No. 9 for the course "Professional Python".

## Description

Profiling and optimization of the employee management system (variant 6):
baseline on Python loops, cProfile, tracemalloc, ThreadPoolExecutor,
ProcessPoolExecutor, NumPy vectorization, Lock synchronization,
lru_cache with cache invalidation and benchmarks with speedup.

## Installation

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

## Profiling

```
python -m employee_manager.profiling
```

## Benchmark

```
python benchmarks/benchmark.py
```

Results: `benchmarks/results/benchmark_results.csv`

## Tests

```
pytest -v
```

## Author

Student: Anna Kolesnik
Group: FEP-33