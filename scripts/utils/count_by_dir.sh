#!/bin/bash

# Define the directory you want to analyze
directory_to_analyze="data/all/"

# Use find to locate all subdirectories
subdirs=($(find "$directory_to_analyze" -type d))

# Iterate through subdirectories and count files
for subdir in "${subdirs[@]}"; do
  file_count=$(find "$subdir" -type f | wc -l)
  echo "$subdir: $file_count files"
done
