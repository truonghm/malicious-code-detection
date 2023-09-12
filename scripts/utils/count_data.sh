#!/bin/bash

# Function to count files in directories with specific names
count_files() {
  local pattern="$1"
  find data -type d -name "all" -exec find {} -type d -iname "$pattern" \; | while read dir; do find "$dir" -maxdepth 1 -type f; done | wc -l
}

# Count files in directories named "badHTML" or "badjs" within "all" subdirectory
bad_count=$(count_files "badHTML")
bad_count=$((bad_count + $(count_files "badjs")))
bad_count=$((bad_count + $(count_files "malware")))
bad_count=$((bad_count + $(count_files "defacement")))
bad_count=$((bad_count + $(count_files "phishing")))

# Count files in directories named "goodHTML" or "goodjs" within "all" subdirectory
good_count=$(count_files "goodHTML")
good_count=$((good_count + $(count_files "goodjs")))
good_count=$((good_count + $(count_files "benign")))

# Print the result
echo "# code snippets labeled as \"good\": $good_count"
echo "# code snippets labeled as \"bad\": $bad_count"
