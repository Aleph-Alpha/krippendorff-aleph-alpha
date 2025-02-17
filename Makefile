.PHONY: check mypy uv

check: mypy uv

mypy:
	mypy src

uv:
	uv run all
