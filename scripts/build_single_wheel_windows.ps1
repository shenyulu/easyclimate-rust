#!/usr/bin/env pwsh

param(
    [string]$PythonPath = "",
    [string]$BuildProfile = "release"
)

# Set error handling
$ErrorActionPreference = "Stop"

# Find Python interpreter
if ([string]::IsNullOrEmpty($PythonPath)) {
    Write-Host "🔍 Auto-searching for Python interpreter..." -ForegroundColor Yellow
    $PythonPath = python -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Unable to automatically find Python interpreter, please specify -PythonPath parameter"
        exit 1
    }
}

# Verify Python version and path
Write-Host "🐍 Using Python: $PythonPath" -ForegroundColor Yellow
& $PythonPath -c "import sys; print('Python version:', sys.version)"

# Check if maturin is installed
Write-Host "🔧 Checking maturin..." -ForegroundColor Yellow
& $PythonPath -c "import maturin; print('maturin version:', maturin.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ maturin not installed, installing now..." -ForegroundColor Red
    & $PythonPath -m pip install maturin
}

# Clean previous builds
if (Test-Path "target\wheels") {
    Write-Host "🧹 Cleaning previous builds..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "target\wheels"
    Remove-Item -Recurse -Force "python\easyclimate_rust\_easyclimate_rust*"
}

# Build configuration
$CargoArgs = @()
if ($BuildProfile -eq "release") {
    $CargoArgs = @("--release")
    Write-Host "🔧 Building in release mode..." -ForegroundColor Green
} else {
    Write-Host "🔧 Building in development mode..." -ForegroundColor Blue
}

# Build wheel using maturin
Write-Host "📦 Starting wheel package build..." -ForegroundColor Yellow
try {
    & $PythonPath -m maturin build @CargoArgs --interpreter $PythonPath --out target/wheels
    
    if ($LASTEXITCODE -eq 0) {
        $wheelFile = Get-ChildItem "target\wheels\*.whl" | Select-Object -First 1
        if ($wheelFile) {
            Write-Host "✅ Build successful!" -ForegroundColor Green
            Write-Host "📦 Wheel package location: $($wheelFile.FullName)" -ForegroundColor Cyan
            
            # Display package information - using safer method
            $wheelPath = $wheelFile.FullName
            Write-Host "`n📊 Package information:" -ForegroundColor Yellow
            Write-Host "   File: $([System.IO.Path]::GetFileName($wheelPath))"
            Write-Host "   Size: $([math]::Round($wheelFile.Length / 1MB, 2)) MB"
            Write-Host "   Path: $wheelPath"
            
            # Try to install and test
            Write-Host "`n🧪 Testing installation..." -ForegroundColor Yellow
            & $PythonPath -m pip install --force-reinstall $wheelPath
            
            # Simple import test
            Write-Host "🧪 Testing import..." -ForegroundColor Yellow
            & $PythonPath -c "
try:
    import easyclimate_rust
    print('✅ Module imported successfully')
    version = getattr(easyclimate_rust, '__version__', 'unknown')
    print(f'   Version: {version}')

    print('🎉 All tests passed!')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"
        } else {
            Write-Error "❌ No generated wheel file found"
            exit 1
        }
    } else {
        Write-Error "❌ Build failed"
        exit 1
    }
} catch {
    Write-Error "❌ Error occurred during build process: $_"
    exit 1
}

Write-Host "`n✨ Build process completed!" -ForegroundColor Green