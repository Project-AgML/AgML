
SHELL := bash

version := 0.7.3

src.python := $(shell find ./agml -type f -name "*.py" || :)
test.python := $(shell find ./tests -type f -name "*.py" || :)

dist.dir := dist
build.wheel := $(dist.dir)/agml-$(version).tar.gz



.PHONY: help
help: ## Print the help screen.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":|:[[:space:]].*?##"}; {printf "\033[36m%-30s\033[0m %s\n", $$2, $$3}'


$(build.wheel): $(src.python)
	uv build -o $(dist.dir)

build: $(build.wheel) ## Build the distribution wheel.

test: $(test.python) $(src.python) # Run tests
	uv run pytest -c=config/pytest.ini $(test.python)