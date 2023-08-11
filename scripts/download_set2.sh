#!/bin/bash

if [[ "$1" == "--redownload" || ! -d "data/set2" ]]; then
    cd data
    git clone https://github.com/zznkiss666/JavaScript_Datasets.git

    mv JavaScript_Datasets set2

    cd set2
    mkdir all
    cp -r badjs all
    cp -r goodjs all

    find . -maxdepth 1 -mindepth 1 -type d ! -name all -exec rm -rf {} \;
    find . -maxdepth 1 -type f -exec rm -f {} \;

    echo "set2 data downloaded successfully!"
else
    echo "set2 directory already exists. Skipping download."
fi
