#!/bin/bash

# Check for at least one argument (the directory path)
if [ $# -lt 1 ]; then
  echo "Usage: $0 <directory> [depth] [--files]"
  exit 1
fi

# Get the directory path from the first argument
dir_path="$1"

# Check if the directory exists
if [ ! -d "$dir_path" ]; then
  echo "Error: Directory '$dir_path' does not exist."
  exit 1
fi

# Get the depth from the second argument, if provided
depth=""
if [ -n "$3" ] && [[ "$3" =~ ^[0-9]+$ ]]; then
  depth="--max-depth=$3"
fi

# Check if the third argument is the flag to include files
include_files="-type d" # Default to showing only directories
if [ "$2" == "--files" ]; then
  include_files="" # Include files by omitting the type restriction
fi

# Use the 'find' command to generate the tree, with the specified depth and file inclusion
find "$dir_path" $depth $include_files -print | sed -e "s;[^/]*/;|__;g;s;__|; |;g"

echo "Tree of directory '$dir_path' generated successfully!"
