#!/bin/bash

set -e

# Universal build script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
PYTHON_PATH="${1:-}"
PROFILE="${2:-release}"
TARGET_DIR="$PROJECT_DIR/target/wheels"

echo "🚀 Starting universal build process..."

# Auto-detect Python
if [ -z "$PYTHON_PATH" ]; then
    if command -v python &> /dev/null; then
        PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
    elif command -v python3 &> /dev/null; then
        PYTHON_PATH=$(which python3)
    else
        echo "❌ Python interpreter not found"
        exit 1
    fi
fi

echo "🐍 Python path: $PYTHON_PATH"
echo "🔧 Build profile: $PROFILE"

# Check and install maturin
if ! $PYTHON_PATH -c "import maturin" 2>/dev/null; then
    echo "📦 Installing maturin..."
    $PYTHON_PATH -m pip install maturin
fi

# Clean build directory
if [ -d "$TARGET_DIR" ]; then
    echo "🧹 Cleaning build directory..."
    rm -rf "$TARGET_DIR"
fi

# Execute build
cd "$PROJECT_DIR"
echo "📦 Starting build process..."

BUILD_ARGS=()
if [ "$PROFILE" = "release" ]; then
    BUILD_ARGS=("--release")
fi

if $PYTHON_PATH -m maturin build "${BUILD_ARGS[@]}" --interpreter "$PYTHON_PATH" --out "$TARGET_DIR"; then
    WHEEL_FILE=$(find "$TARGET_DIR" -name "*.whl" | head -n 1)
    if [ -n "$WHEEL_FILE" ]; then
        echo ""
        echo "✅ Build successful!"
        echo "📊 Build information:"
        echo "   File: $(basename "$WHEEL_FILE")"
        echo "   Size: $(du -h "$WHEEL_FILE" | cut -f1)"
        echo "   Path: $WHEEL_FILE"
        echo "   Profile: $PROFILE"
        
        # Verify build by installing and testing
        echo ""
        echo "🧪 Verifying build..."
        echo "   Installing package..."
        $PYTHON_PATH -m pip install --force-reinstall "$WHEEL_FILE"
        
        echo "   Testing import..."
        $PYTHON_PATH -c "
import sys
try:
    import easyclimate_rust as ecl
    print('✅ Import successful')
    print(f'   Version: {getattr(ecl, \"__version__\", \"unknown\")}')
    
    # Test basic functionality if available
    if hasattr(ecl, 'calculate_climate_index'):
        result = ecl.calculate_climate_index(25.0, 60.0)
        print(f'   Function test: calculate_climate_index(25.0, 60.0) = {result}')
    
    print('🎉 Build verification passed!')
except Exception as e:
    print(f'❌ Verification failed: {e}')
    sys.exit(1)
"
    fi
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✨ Build process completed successfully!"