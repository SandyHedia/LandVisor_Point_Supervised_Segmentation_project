# Variables
PYTHON = venv/bin/python
PIP = venv/bin/pip
CONFIG = configs/base_config.yaml

.PHONY: setup train eval test clean help

## setup      : Install dependencies and prepare virtual environment
setup:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## train      : Start the training pipeline using the base config
train:
	$(PYTHON) scripts/train.py

## eval       : Run evaluation on the test set and generate report
eval:
	$(PYTHON) scripts/evaluate.py

## test       : Run unit tests for custom losses and data logic
test:
	$(PYTHON) -m unittest discover tests

## clean      : Remove temporary files, pycache, and checkpoints
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -f *.pth
	rm -f *.csv
	rm -f *.png

## help       : Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^##' Makefile | sed -e 's/## //'
	