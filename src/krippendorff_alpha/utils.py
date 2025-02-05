import subprocess


def run_ruff() -> None:
    subprocess.run(["ruff", "check"], check=True)


def run_format() -> None:
    subprocess.run(["ruff", "format"], check=True)


def run_pytest() -> None:
    subprocess.run(["pytest"], check=True)


def run_mypy() -> None:
    subprocess.run(["mypy", "src", "tests"], check=True)


def run_all() -> None:
    run_ruff()
    run_format()
    run_pytest()
    run_mypy()
