.ONESHELL:
SHELL = /bin/bash
CONDA_ENV_PATH=.conda/m1
CONDA_HOME_PATH=$(HOME)/miniconda3

download:
	./scripts/download_set1.sh
	./scripts/download_set2.sh

count:
	./scripts/count_data.sh

tree:
	./scripts/gen_tree.sh data

env: 
	source $(CONDA_HOME_PATH)/bin/activate; conda create -p $(CONDA_ENV_PATH) --no-default-packages --no-deps python=3.10 -y; conda env update -p $(CONDA_ENV_PATH) --file environment.yml

env-reset:
	rm -rf $(CONDA_ENV_PATH)
	make env

format:
	black src --config pyproject.toml
	ruff src --fix --config pyproject.toml

## Run checks (ruff + test)
check:
	ruff check src --config pyproject.toml
	black --check src --config pyproject.toml

type:
	mypy src --config-file pyproject.toml