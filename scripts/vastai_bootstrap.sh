#!/bin/bash

# install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/install.python-poetry.org/main/install-poetry.py | python3

poetry --version
# add poetry to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# create environment
make env

conda activate .conda/m1

poetry install --no-root