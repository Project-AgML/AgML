
SHELL := bash

version := 0.7.0

src.python := $(shell find ./agml -type f -name "*.py" || :)
test.python := $(shell find ./tests -type f -name "*.py" || :)

dist.dir := dist
build.wheel := $(dist.dir)/agml-$(version).tar.gz



.PHONY: help
help: ## Print the help screen.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":|:[[:space:]].*?##"}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


$(build.wheel): $(src.python) ## Build wheels
	uv build -o $(dist.dir)

build: $(build.wheel) ## Build the distribution wheel.

test: $(test.python) $(src.python) ## Run tests
	uv run pytest -c=config/pytest.ini $(test.python)


.PHONY: format
format: ## Run format
	uv run ruff format --config=config/ruff.toml $(src.python) $(test.python)

.PHONY: check-Format
check-format: ## Check format
	uv run ruff format --check --config=config/ruff.toml $(src.python) $(test.python)


.PHONY: lint-fix
lint-fix: ## Fix Lint issues
	uv run ruff check --fix --config=config/ruff.toml $(src.python) $(test.python)

.PHONY: lint
lint: ## Run Code Linter
	uv run ruff check --config=config/ruff.toml $(src.python) $(test.python)
