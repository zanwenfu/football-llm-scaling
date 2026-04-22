.PHONY: help install install-train test lint format analyze figures contradictions demo all clean

PY ?= python3
PYTHONPATH := src

help:
	@echo "Targets:"
	@echo "  install        pip install the package (analysis-only deps)"
	@echo "  install-train  pip install with training / inference deps"
	@echo "  test           run the pytest suite"
	@echo "  lint           ruff check"
	@echo "  format         ruff format + isort"
	@echo "  analyze        rebuild results/tables/ from results/raw/"
	@echo "  figures        rebuild results/figures/ from results/raw/"
	@echo "  contradictions rebuild results/examples/ from results/raw/"
	@echo "  demo           run the structured-output rescue demo"
	@echo "  all            analyze + figures + contradictions + demo"
	@echo "  clean          remove build artifacts and caches"

install:
	$(PY) -m pip install -e .

install-train:
	$(PY) -m pip install -e '.[train,dev]'

test:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest tests/

lint:
	$(PY) -m ruff check src tests scripts

format:
	$(PY) -m ruff format src tests scripts

analyze:
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/03_analyze_results.py \
		--raw-dir results/raw --tables-dir results/tables

figures:
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/04_make_figures.py \
		--raw-dir results/raw --figures-dir results/figures

contradictions:
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/05_dump_contradictions.py \
		--raw-dir results/raw --out results/examples/example_contradictions.md

demo:
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/demo_structured_rescue.py

all: analyze figures contradictions demo

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
