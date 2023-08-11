#!/bin/bash

if [[ "$1" == "--redownload" || ! -d "data/set2" ]]; then
    cd data
    git clone https://github.com/zznkiss666/JavaScript_Datasets.git

    mv JavaScript_Datasets set2

    mkdir all

    cp -r set2/badjs all
    cp -r set2/goodjs all

    cd set2

    find . -maxdepth 1 -mindepth 1 -type d ! -name badjs ! -name goodjs -exec rm -rf {} \;
    find . -maxdepth 1 -type f -exec rm -f {} \;

    echo "set2 data downloaded successfully!"
else
    echo "set2 directory already exists. Skipping download."
fi
