#!/bin/bash
set -e

# Usage check
if [ -z "$1" ]; then
  echo "Usage: $0 <run_number>"
  exit 1
fi

RUN_NUMBER="$1"
TOML_FILE="pyproject.toml"

# Extract base version from pyproject.toml
BASE_VERSION=$(grep -E '^version *= *' "$TOML_FILE" | sed -E 's/version *= *"([^"]+)"/\1/')

# Compose full version string
FULL_VERSION="${BASE_VERSION}.${RUN_NUMBER}"

# Export as GitHub Actions env variable
echo "VERSION_WITH_BUILD=$FULL_VERSION" >> $GITHUB_ENV

# Optional: log version
echo "Extracted version: $FULL_VERSION"