#!/bin/bash

if [[ "$1" == "--redownload" || ! -d "data/set2" ]]; then
    cd data
    git clone https://github.com/zznkiss666/JavaScript_Datasets.git

    mv JavaScript_Datasets set2

    mkdir all

    cp -r set2/badjs all/misc2/badjs
    cp -r set2/goodjs all/misc2/goodjs

    rm -rf set2

    echo "set2 data downloaded successfully!"
else
    echo "set2 directory already exists. Skipping download."
fi
