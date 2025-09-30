#!/bin/bash

# This scripts is for the github CI/CD so it can change the version number when upload to test.pypi.org
# so we do not get build conflicts

# Check if the number of arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <string_to_append>"
    exit 1
fi

# Find pyproject.toml in the current directory
file="pyproject.toml"

# Check if the file exists
if [ -f "$file" ]; then
    # Find the line that starts with "version ="
    line=$(grep -n "^version = " "$file" | cut -d: -f1)

    # Append the input argument to the version
    sed -i "${line}s/\"$/\.$1\"/" "$file"

    echo "Successfully appended '$1' to the version in $file"
else
    echo "File pyproject.toml not found in the current directory."
    exit 1
fi