import subprocess  # nosec
import sys
import argparse


def run_pipeline_step(commands: list[str]) -> None:
    result = subprocess.run(commands, check=True)  # nosec
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_ruff() -> None:
    run_pipeline_step(["ruff", "check"])


def run_format() -> None:
    run_pipeline_step(["ruff", "format"])


def run_pytest() -> None:
    parser = argparse.ArgumentParser()
    args, extra_args = parser.parse_known_args()
    commands = ["pytest", "-n", "10"]
    for arg in extra_args:
        commands.append(arg)
    run_pipeline_step(commands)


def run_mypy() -> None:
    run_pipeline_step(["mypy", "src", "tests"])


def run_all() -> None:
    run_format()
    run_ruff()
    run_pytest()
    run_mypy()
