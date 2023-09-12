#!/bin/bash

if [[ "$1" == "--redownload" || ! -d "data/set1" ]]; then
	cd data

	git clone https://github.com/njuhxc/MJDetector.git
	mv MJDetector set1
	cd set1

	mv TestingSet test
	mv TrainingSet train

	cd ..
	mkdir all
	cp -r set1/test/* all/misc
	cp -r set1/train/* all/misc
	rm -rf set1

	echo "set1 data downloaded successfully!"
else
	echo "set1 directory already exists. Skipping download."
fi
