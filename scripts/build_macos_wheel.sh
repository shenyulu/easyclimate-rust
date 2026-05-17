#!/bin/bash

set -euo pipefail

PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13" "3.14")
BUILD_PROFILE="release"

clean_build_artifacts() {
    echo "🧹 Cleaning old build artifacts..."

    find . \
        \( -path "./target" -o -path "./.git" -o -path "./.venv" \) -prune -o \
        -type f \( -name "*.so" -o -name "*.dylib" \) -print -delete

    rm -rf ./python/__pycache__
    rm -rf ./python/easyclimate_rust/__pycache__
    rm -rf ./target/maturin
    rm -rf ./.venv

    mkdir -p dist

    echo "✅ Cleanup completed"
}

ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "❌ uv is not installed. Please install uv first:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

build_wheels() {
    for version in "${PYTHON_VERSIONS[@]}"; do
        echo
        echo "========================================"
        echo "Building macOS wheel for Python ${version}..."
        echo "========================================"

        clean_build_artifacts

        uv run --python "${version}" --with maturin maturin build --${BUILD_PROFILE} -o dist/

        echo "✅ Completed wheel for Python ${version}"
    done
}

show_results() {
    echo
    echo "🎉 All macOS wheels built successfully!"
    echo
    echo "📦 Generated wheels:"
    ls -1 dist/*.whl
}

ensure_uv
clean_build_artifacts
build_wheels
show_results
