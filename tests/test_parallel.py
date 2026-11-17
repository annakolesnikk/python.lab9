from pathlib import Path

import numpy as np
import pytest

from employee_manager.analytics import (
    PayrollAnalytics,
    calculate_projections_numpy,
    calculate_projections_sequential,
    department_statistics_numpy,
    department_statistics_python,
    generate_employees,
    payroll_statistics_numpy,
    payroll_statistics_python,
    salary_ranking,
)
from employee_manager.parallel import (
    ImportStatistics,
    calculate_projections_processes,
    calculate_projections_threads,
    export_by_department,
    load_files_sequential,
    load_files_threads,
    race_condition_demo,
)

SALARIES = [30_000.0, 45_000.5, 60_000.0, 75_000.25, 120_000.0]


@pytest.fixture(scope="module")
def employees():
    return generate_employees(2_000)


def test_numpy_statistics_match_python() -> None:
    python_result = payroll_statistics_python(SALARIES)
    numpy_result = payroll_statistics_numpy(SALARIES)

    for key in python_result:
        assert numpy_result[key] == pytest.approx(python_result[key])
    assert python_result["maximum"] == 120_000.0
    assert python_result["total"] == pytest.approx(330_000.75)


def test_department_statistics_match(employees) -> None:
    python_result = department_statistics_python(employees)
    numpy_result = department_statistics_numpy(
        np.asarray([e.department for e in employees]),
        np.asarray([e.salary for e in employees]),
    )

    assert python_result.keys() == numpy_result.keys()
    for department, stats in python_result.items():
        for key in ("count", "total", "maximum", "average"):
            assert numpy_result[department][key] == pytest.approx(stats[key])


def test_salary_ranking(employees) -> None:
    ranking = salary_ranking(employees, top=5)

    assert len(ranking) == 5
    assert ranking[0].salary == max(e.salary for e in employees)
    assert [e.salary for e in ranking] == sorted((e.salary for e in ranking), reverse=True)


def test_threads_and_processes_match_sequential() -> None:
    sequential = calculate_projections_sequential(SALARIES)

    assert calculate_projections_threads(SALARIES, workers=2) == sequential
    assert calculate_projections_processes(SALARIES, workers=2, chunksize=2) == sequential


def test_numpy_projection_matches_sequential() -> None:
    sequential = calculate_projections_sequential(SALARIES)

    assert np.allclose(calculate_projections_numpy(SALARIES), sequential, rtol=1e-9)


def test_parallel_import_matches_sequential(employees, tmp_path: Path) -> None:
    paths = export_by_department(employees, tmp_path)
    statistics = ImportStatistics()

    sequential = load_files_sequential(paths)
    parallel = load_files_threads(paths, workers=4, statistics=statistics)

    assert sorted(e.name for e in parallel) == sorted(e.name for e in sequential)
    assert statistics.rows == len(employees)


def test_lock_prevents_race_condition() -> None:
    unsafe, safe = race_condition_demo(threads=8, increments=200)

    assert safe == 1_600
    assert unsafe <= safe


def test_cache_hit_and_invalidation(employees) -> None:
    analytics = PayrollAnalytics(employees)

    first = analytics.statistics("IT")
    second = analytics.statistics("IT")
    assert second is first
    assert analytics.department_statistics.cache_info().hits == 1

    index = next(i for i, e in enumerate(employees) if e.department == "IT")
    analytics.update_salary(index, 10_000_000)
    updated = analytics.statistics("IT")

    assert updated["maximum"] == 10_000_000
    assert analytics.department_statistics.cache_info().misses == 2


def test_transfer_invalidates_cache(employees) -> None:
    analytics = PayrollAnalytics(employees)
    before = analytics.statistics("HR")["count"]

    index = next(i for i, e in enumerate(employees) if e.department == "IT")
    analytics.transfer(index, "HR")

    assert analytics.statistics("HR")["count"] == before + 1