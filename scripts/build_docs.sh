#!/bin/bash

# traj-dist-rs Documentation Build Script
# This script builds the Sphinx documentation for traj-dist-rs with MyST support

set -e  # Exit immediately if a command exits with a non-zero status

# Project root directory (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Documentation source and build directories
DOCS_SRC_DIR="$PROJECT_ROOT/docs"
DOCS_BUILD_DIR="$PROJECT_ROOT/docs/_build"

# Print status message
print_status() {
    echo ">>> $1"
}

# Check if we're in the correct project directory
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ] || [ ! -f "$PROJECT_ROOT/Cargo.toml" ]; then
    echo "Error: Could not find project root. This script should be run from the traj-dist-rs directory."
    exit 1
fi

print_status "Starting documentation build for traj-dist-rs with MyST support..."

# Check if sphinx-build is available
if ! command -v sphinx-build &> /dev/null; then
    print_status "sphinx-build not found in PATH, trying to use Python module..."
    if ! python -c "import sphinx" &> /dev/null; then
        print_status "Error: Sphinx is not installed. Please install it with: uv add --dev sphinx"
        exit 1
    fi
    SPHINX_BUILD_CMD="python -m sphinx.cmd.build"
else
    SPHINX_BUILD_CMD="sphinx-build"
fi

# Clean previous build if requested
if [ "$1" = "--clean" ] || [ "$1" = "-c" ]; then
    print_status "Cleaning previous build..."
    rm -rf "$DOCS_BUILD_DIR"
    mkdir -p "$DOCS_BUILD_DIR"
fi

# Build the documentation
print_status "Building HTML documentation with MyST support..."
$SPHINX_BUILD_CMD -b html "$DOCS_SRC_DIR" "$DOCS_BUILD_DIR" --fresh-env

print_status "Documentation build completed successfully!"
print_status "HTML documentation is available at: $DOCS_BUILD_DIR/index.html"

# Optional: Open the documentation in a web browser (uncomment if desired)
# if command -v xdg-open &> /dev/null; then
#     print_status "Opening documentation in web browser..."
#     xdg-open "$DOCS_BUILD_DIR/index.html"
# elif command -v open &> /dev/null; then
#     print_status "Opening documentation in web browser..."
#     open "$DOCS_BUILD_DIR/index.html"
# fi

print_status "Build completed at $(date)"