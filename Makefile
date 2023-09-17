.ONESHELL:
SHELL = /bin/bash
CONDA_ENV_PATH=.conda/m1
CONDA_HOME_PATH=$(HOME)/miniconda3

## Download data for training
download:
	./scripts/utils/download_set1.sh
	./scripts/utils/download_set2.sh

count:
	./scripts/utils/count_data.sh

## Generate tree of data folder
tree:
	./scripts/utils/gen_tree.sh data

## Create conda env (python 3.10) using environment.yml
env: 
	source $(CONDA_HOME_PATH)/bin/activate; conda create -p $(CONDA_ENV_PATH) --no-default-packages --no-deps python=3.10 -y; conda env update -p $(CONDA_ENV_PATH) --file environment.yml
	touch .conda/.gitignore
	echo "*" > .conda/.gitignore

bootstrap:
	source ./scripts/utils/vastai_bootstrap.sh
## Remove old conda env and create a new one
env-reset:
	rm -rf $(CONDA_ENV_PATH)
	make env

split:
	python scripts/split_train_test.py --input=data/all/kaggle1,data/all/misc2,data/all/packt --output=data/exp --sample-size=0.2 --train-size=0.8
	PYTHONPATH=$(shell pwd) python scripts/create_train_input.py

tokenize:
	PYTHONPATH=$(shell pwd) python scripts/tokenize_corpus.py --input=data/exp/train_set.csv --output=data/exp
	PYTHONPATH=$(shell pwd) python scripts/tokenize_corpus.py --input=data/exp/test_set.csv --output=data/exp

fasttext:
	PYTHONPATH=$(shell pwd) python scripts/build_fasttext_model.py --input=data/exp/train_set_token_types_corpus.txt --model-dir=models/fasttext_embeddings.bin --no-hierarchical-softmax

PATH_TO_CHECK=./lib/* ./crawler/*
## Format files using black, using pre-commit hooks
format:
	pre-commit run ruff-lint --files $(PATH_TO_CHECK)
	pre-commit run black-format --files $(PATH_TO_CHECK)

## Run checks (ruff + test), using pre-commit hooks
check-format:
	pre-commit run ruff-check --files $(PATH_TO_CHECK)
	pre-commit run black-check --files $(PATH_TO_CHECK)

## Run mypy type checking using pre-commit hook
check-mypy:
	pre-commit run mypy --files $(PATH_TO_CHECK)

## Run all checks (ruff + test + mypy), using pre-commit hooks
check-all:
	pre-commit run --files $(PATH_TO_CHECK)

## crawl urls from the kaggle dataset
crawl:
	export PYTHONPATH=$(shell pwd) && python scripts/utils/crawl_kaggle_dataset.py --skip=108201 --limit=700000 --input=data/malicious_phish.csv --output=data/all/kaggle1 --super_label=goodjs

## render report
render:
	quarto render ./report/index.qmd

## preview report
preview:
	quarto preview ./report/index.qmd --no-watch-inputs --no-browser --port 7733

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')