
SHELL := bash

version := 0.7.0

src.python := $(shell find ./agml -type f -name "*.py" || :)
test.python := $(shell find ./tests -type f -name "*.py" || :)

dist.dir := dist
build.wheel := $(dist.dir)/agml-$(version).tar.gz


install: setup ## Install dependencies including development ones
	uv sync --dev

setup: ## setup env
	uv venv .venv


.PHONY: help
help: ## Print the help screen.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":|:[[:space:]].*?##"}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


$(build.wheel): $(src.python) ## Build wheels
	uv build -o $(dist.dir)

build: $(build.wheel) ## Build the distribution wheel.

test: $(test.python) $(src.python) ## Run tests
	uv run pytest -c=config/pytest.ini $(test.python)


# Quality Checks

.PHONY: checks
checks: check-format lint check-types docstring-coverage


.PHONY: check-types
check-types: ## Run mypy to check type definitions.
	uv run mypy --config=config/mypy.ini $(src.python) $(test.python)


.PHONY: check-format
check-format: ## Check ruff format
	uv run ruff format --check --config=config/ruff.toml $(src.python) $(test.python)


.PHONY: lint
lint: ## Run ruff Code Linter
	uv run ruff check --config=config/ruff.toml $(src.python) $(test.python)

.PHONY:docstring-coverage
docstring-coverage: ## Compute docstring coverage
	uv run interrogate -c config/interrogate.toml .

# Quality fixes

.PHONY: lint-fix
lint-fix: ## Fix ruff Lint issues
	uv run ruff check --fix --config=config/ruff.toml $(src.python) $(test.python)


.PHONY: format
format: ## Run ruff format (Includes sorting of imports)
	uv run ruff check --select I --config=config/ruff.toml --fix
	uv run ruff format --config=config/ruff.toml $(src.python) $(test.python)
