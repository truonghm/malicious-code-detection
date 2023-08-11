#!/bin/bash

if [[ "$1" == "--redownload" || ! -d "data/set1" ]]; then
	cd data

	git clone https://github.com/njuhxc/MJDetector.git
	mv MJDetector set1
	cd set1

	mv TestingSet test
	mv TrainingSet train

	find . -maxdepth 1 -type f -exec rm -f {} \;
	find . -maxdepth 1 -mindepth 1 -type d ! -name test ! -name train -exec rm -rf {} \;

	mkdir all
	cp -r test/* all
	cp -r train/* all

	echo "set1 data downloaded successfully!"
else
	echo "set1 directory already exists. Skipping download."
fi
