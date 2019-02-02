.PHONY: all install uninstall lint format test clean FORCE

all: test FORCE

install: FORCE
		pip install -e .[dev]

uninstall: FORCE
		pip uninstall gpyro

lint: FORCE
		flake8

format: FORCE
		isort -rc .

test: lint FORCE
		pytest

clean: FORCE
		git clean -dfx -e gpyro.egg-info

FORCE:
