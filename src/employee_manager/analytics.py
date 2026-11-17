import math
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DEPARTMENTS = ["Sales", "IT", "Finance", "HR", "Marketing", "Legal", "Support", "Logistics"]
POSITIONS = ["Junior", "Middle", "Senior", "Lead", "Manager"]


@dataclass(frozen=True, slots=True)
class EmployeeRecord:
    name: str
    position: str
    department: str
    salary: float


def generate_employees(count: int, seed: int = 42) -> list[EmployeeRecord]:
    rng = random.Random(seed)
    return [
        EmployeeRecord(
            name=f"Employee {index}",
            position=rng.choice(POSITIONS),
            department=rng.choice(DEPARTMENTS),
            salary=round(rng.uniform(20_000, 150_000), 2),
        )
        for index in range(count)
    ]


# ---------- baseline: pure Python loops ----------

def payroll_statistics_python(salaries: list[float]) -> dict[str, float]:
    count = len(salaries)
    total = 0.0
    maximum = salaries[0]
    minimum = salaries[0]
    for salary in salaries:
        total += salary
        if salary > maximum:
            maximum = salary
        if salary < minimum:
            minimum = salary
    average = total / count
    variance = sum((salary - average) ** 2 for salary in salaries) / count
    return {
        "total": total,
        "average": average,
        "maximum": maximum,
        "minimum": minimum,
        "std": math.sqrt(variance),
    }


def department_statistics_python(
    employees: list[EmployeeRecord],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for employee in employees:
        stats = result.setdefault(
            employee.department, {"count": 0, "total": 0.0, "maximum": 0.0}
        )
        stats["count"] += 1
        stats["total"] += employee.salary
        stats["maximum"] = max(stats["maximum"], employee.salary)
    for stats in result.values():
        stats["average"] = stats["total"] / stats["count"]
    return result


def salary_ranking(employees: list[EmployeeRecord], top: int = 10) -> list[EmployeeRecord]:
    return sorted(employees, key=lambda employee: employee.salary, reverse=True)[:top]


# ---------- NumPy ----------

def payroll_statistics_numpy(salaries: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(salaries, dtype=np.float64)
    return {
        "total": float(array.sum()),
        "average": float(array.mean()),
        "maximum": float(array.max()),
        "minimum": float(array.min()),
        "std": float(array.std()),
    }


def department_statistics_numpy(
    departments: np.ndarray,
    salaries: np.ndarray,
) -> dict[str, dict[str, float]]:
    names, codes = np.unique(departments, return_inverse=True)
    counts = np.bincount(codes)
    totals = np.bincount(codes, weights=salaries)
    maximums = np.full(len(names), -np.inf)
    np.maximum.at(maximums, codes, salaries)
    return {
        str(name): {
            "count": int(counts[i]),
            "total": float(totals[i]),
            "maximum": float(maximums[i]),
            "average": float(totals[i] / counts[i]),
        }
        for i, name in enumerate(names)
    }


# ---------- CPU-bound operation ----------

PROJECTION_YEARS = 200


def salary_projection(salary: float) -> float:
    """Simulated long-term salary projection: raises, inflation and taxes per period."""
    value = salary
    for year in range(PROJECTION_YEARS):
        raise_rate = 0.03 + 0.01 * math.sin(year)
        tax = 0.18 + 0.015 * math.cos(value / 10_000)
        value = value * (1 + raise_rate) * (1 - tax) / (1 - 0.18) / 1.03
        value = math.sqrt(value * salary)
    return value


def calculate_projections_sequential(salaries: list[float]) -> list[float]:
    return [salary_projection(salary) for salary in salaries]


def calculate_projections_numpy(salaries: list[float] | np.ndarray) -> np.ndarray:
    base = np.asarray(salaries, dtype=np.float64)
    value = base.copy()
    for year in range(PROJECTION_YEARS):
        raise_rate = 0.03 + 0.01 * math.sin(year)
        tax = 0.18 + 0.015 * np.cos(value / 10_000)
        value = value * (1 + raise_rate) * (1 - tax) / (1 - 0.18) / 1.03
        value = np.sqrt(value * base)
    return value


# ---------- caching with invalidation ----------

class PayrollAnalytics:
    """Department statistics with lru_cache; data version is part of the cache key."""

    def __init__(self, employees: list[EmployeeRecord]) -> None:
        self._employees = list(employees)
        self.version = 0
        self.department_statistics = lru_cache(maxsize=128)(self._department_statistics)

    def _department_statistics(self, department: str, version: int) -> dict[str, float]:
        salaries = [e.salary for e in self._employees if e.department == department]
        return {
            "count": len(salaries),
            "total": sum(salaries),
            "average": sum(salaries) / len(salaries) if salaries else 0.0,
            "maximum": max(salaries, default=0.0),
        }

    def statistics(self, department: str) -> dict[str, float]:
        return self.department_statistics(department, self.version)

    def update_salary(self, index: int, salary: float) -> None:
        old = self._employees[index]
        self._employees[index] = EmployeeRecord(old.name, old.position, old.department, salary)
        self.version += 1  # invalidation: new cache key

    def transfer(self, index: int, department: str) -> None:
        old = self._employees[index]
        self._employees[index] = EmployeeRecord(old.name, old.position, department, old.salary)
        self.version += 1