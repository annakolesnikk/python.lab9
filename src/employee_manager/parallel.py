import csv
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from employee_manager.analytics import EmployeeRecord, salary_projection


# ---------- CPU-bound: threads vs processes ----------

def calculate_projections_threads(salaries: list[float], workers: int = 4) -> list[float]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(salary_projection, salaries))


def calculate_projections_processes(
    salaries: list[float],
    workers: int = 4,
    chunksize: int = 500,
) -> list[float]:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(salary_projection, salaries, chunksize=chunksize))


# ---------- I/O-bound: parallel import of department files ----------

def export_by_department(employees: list[EmployeeRecord], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[EmployeeRecord]] = {}
    for employee in employees:
        groups.setdefault(employee.department, []).append(employee)

    paths = []
    for department, rows in groups.items():
        path = directory / f"{department}.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["name", "position", "department", "salary"])
            writer.writerows((e.name, e.position, e.department, e.salary) for e in rows)
        paths.append(path)
    return sorted(paths)


class ImportStatistics:
    """Shared mutable state updated from several threads."""

    def __init__(self) -> None:
        self.rows = 0
        self._lock = threading.Lock()

    def add_safe(self, count: int) -> None:
        with self._lock:  # critical section is as short as possible
            self.rows += count

    def add_unsafe(self, count: int) -> None:
        current = self.rows
        time.sleep(0)  # lets another thread run between read and write
        self.rows = current + count


def load_file(
    path: Path,
    statistics: ImportStatistics | None = None,
    latency: float = 0.0,
) -> list[EmployeeRecord]:
    time.sleep(latency)  # simulated slow storage / network disk
    with path.open(encoding="utf-8", newline="") as file:
        rows = [
            EmployeeRecord(row["name"], row["position"], row["department"], float(row["salary"]))
            for row in csv.DictReader(file)
        ]
    if statistics is not None:
        statistics.add_safe(len(rows))
    return rows


def load_files_sequential(paths: list[Path], latency: float = 0.0) -> list[EmployeeRecord]:
    result: list[EmployeeRecord] = []
    for path in paths:
        result.extend(load_file(path, latency=latency))
    return result


def load_files_threads(
    paths: list[Path],
    workers: int = 8,
    latency: float = 0.0,
    statistics: ImportStatistics | None = None,
) -> list[EmployeeRecord]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = executor.map(lambda path: load_file(path, statistics, latency), paths)
        return [employee for part in parts for employee in part]


# ---------- race condition demo ----------

def race_condition_demo(threads: int = 8, increments: int = 1_000) -> tuple[int, int]:
    unsafe = ImportStatistics()
    safe = ImportStatistics()

    def work() -> None:
        for _ in range(increments):
            unsafe.add_unsafe(1)
            safe.add_safe(1)

    workers = [threading.Thread(target=work) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return unsafe.rows, safe.rows