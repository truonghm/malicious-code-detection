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
