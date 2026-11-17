import cProfile
import pstats
import tracemalloc

import numpy as np

from employee_manager.analytics import (
    EmployeeRecord,
    calculate_projections_sequential,
    department_statistics_python,
    generate_employees,
    payroll_statistics_python,
    salary_ranking,
)

DATA_SIZE = 200_000
PROJECTION_SIZE = 5_000


def baseline_workload(employees: list[EmployeeRecord]) -> None:
    salaries = [employee.salary for employee in employees]
    payroll_statistics_python(salaries)
    department_statistics_python(employees)
    salary_ranking(employees)
    calculate_projections_sequential(salaries[:PROJECTION_SIZE])


def run_profile() -> None:
    employees = generate_employees(DATA_SIZE)

    profiler = cProfile.Profile()
    profiler.enable()
    baseline_workload(employees)
    profiler.disable()

    statistics = pstats.Stats(profiler)
    statistics.sort_stats("cumulative")
    statistics.print_stats(12)


def memory_test() -> None:
    tracemalloc.start()
    employees = generate_employees(DATA_SIZE)
    _, peak_records = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    salaries_list = [employee.salary for employee in employees]
    _, peak_list = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    salaries_array = np.asarray(salaries_list, dtype=np.float64)
    _, peak_array = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nDataset: {DATA_SIZE} employees")
    print(f"Peak memory, list[EmployeeRecord]: {peak_records / 1024 / 1024:8.2f} MB")
    print(f"Peak memory, list[float] salaries: {peak_list / 1024 / 1024:8.2f} MB")
    print(f"Peak memory, NumPy array:          {peak_array / 1024 / 1024:8.2f} MB")
    print(f"NumPy array nbytes:                {salaries_array.nbytes / 1024 / 1024:8.2f} MB")


if __name__ == "__main__":
    run_profile()
    memory_test()