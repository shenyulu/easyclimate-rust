#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_PATH="${1:-}"
BUILD_PROFILE="${2:-release}"
TARGET_DIR="$PROJECT_DIR/target/wheels"

echo "🚀 Starting macOS single-environment build process..."

if [ -z "$PYTHON_PATH" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_PATH="$(python3 -c 'import sys; print(sys.executable)')"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_PATH="$(python -c 'import sys; print(sys.executable)')"
    else
        echo "❌ Python interpreter not found in current environment"
        exit 1
    fi
fi

echo "🐍 Python path: $PYTHON_PATH"
echo "🔧 Build profile: $BUILD_PROFILE"

if ! "$PYTHON_PATH" -c "import maturin" >/dev/null 2>&1; then
    echo "📦 Installing maturin..."

    PIP_INSTALL_ARGS=(install maturin)
    PIP_INDEX_URL="$("$PYTHON_PATH" -m pip config get global.index-url 2>/dev/null || true)"

    if [[ "$PIP_INDEX_URL" == http://* ]]; then
        HTTPS_INDEX_URL="https://${PIP_INDEX_URL#http://}"
        echo "⚠️ Detected insecure pip index-url: $PIP_INDEX_URL"
        echo "🔁 Retrying with HTTPS index-url: $HTTPS_INDEX_URL"
        PIP_INSTALL_ARGS+=(--index-url "$HTTPS_INDEX_URL")
    fi

    "$PYTHON_PATH" -m pip "${PIP_INSTALL_ARGS[@]}"
fi

if [ -d "$TARGET_DIR" ]; then
    echo "🧹 Cleaning old wheel output..."
    rm -rf "$TARGET_DIR"
fi

cd "$PROJECT_DIR"

BUILD_ARGS=()
if [ "$BUILD_PROFILE" = "release" ]; then
    BUILD_ARGS+=("--release")
fi

echo "📦 Building wheel with current Python environment..."
"$PYTHON_PATH" -m maturin build "${BUILD_ARGS[@]}" --interpreter "$PYTHON_PATH" --out "$TARGET_DIR"

WHEEL_FILE="$(find "$TARGET_DIR" -maxdepth 1 -name "*.whl" | head -n 1)"

if [ -z "$WHEEL_FILE" ]; then
    echo "❌ No wheel file was generated"
    exit 1
fi

echo
echo "✅ Build successful!"
echo "📊 Build information:"
echo "   File: $(basename "$WHEEL_FILE")"
echo "   Size: $(du -h "$WHEEL_FILE" | cut -f1)"
echo "   Path: $WHEEL_FILE"
echo "   Profile: $BUILD_PROFILE"

echo
echo "🧪 Verifying wheel installation..."
"$PYTHON_PATH" -m pip install --force-reinstall "$WHEEL_FILE"

echo "🧪 Testing import..."
"$PYTHON_PATH" -c '
import sys
try:
    import easyclimate_rust as ecl
    print("✅ Import successful")
    print(f"   Version: {getattr(ecl, "__version__", "unknown")}")
    print("🎉 Build verification passed!")
except Exception as exc:
    print(f"❌ Verification failed: {exc}")
    sys.exit(1)
'

echo
echo "✨ macOS single-environment build completed successfully!"
