<img src="https://github.com/shenyulu/easyclimate-rust/blob/main/docs/logo/easyclimate_rust_logo_mini.png?raw=true" alt="easyclimate-rust">

<h2 align="center">The Rust backend of easyclimate</h2>

![PyPI - Version](https://img.shields.io/pypi/v/easyclimate-rust)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/easyclimate-rust)
![PyPI - Downloads](https://img.shields.io/pypi/dm/easyclimate-rust)

<div align="center">
<center>English / <a href = "readme/README_zh_CN.md">简体中文</a> / <a href = "readme/README_ja_JP.md">日本語</a></center>
</div>

## 🤗 What is easyclimate-rust?

**easyclimate-rust** is a high-performance backend library written in Rust, designed to handle computationally intensive tasks for
[easyclimate](https://github.com/shenyulu/easyclimate).

By leveraging Rust’s **high performance**, **memory safety**, and **zero-cost abstractions**,
easyclimate-rust enables the Python front-end to provide a clean, user-friendly interface for climate data analysis,
while ensuring scalability and efficiency for large datasets and complex diagnostics.

> 🚨 **Project Status: Actively Developing** 🚨
>
> This package is under rapid development.
> APIs (functions, classes, and interfaces) are **not yet stable** and may change without backward compatibility.
> Use with caution in production environments.

## 😯 Installation

The `easyclimate-rust` package can be installed via the Python package manager
[pip](https://pip.pypa.io/en/stable/getting-started/):

```bash
pip install easyclimate-rust
```

## ✨ Requirements

* **Python** ≥ 3.10
* **NumPy** ≥ 1.24.3
  *(Required only at runtime for the prebuilt wheel)*

## 🔧 Build Instructions

### Windows

1. Install Rust
   👉 [Set up your dev environment on Windows for Rust](https://learn.microsoft.com/en-us/windows/dev-environment/rust/setup)
2. Install `uv`:

```powershell
winget install uv
```

3. Run the build script from the project root:

```powershell
.\scripts\build_manywindows_wheel.ps1
```

4. The generated wheel file will be located in the `dist/` directory.

### Linux

1. Install Docker on your system.
2. Run the build script on a Linux host:

```bash
./scripts/build_manylinux_wheel.sh
```

The resulting wheel will also be placed in the `dist/` directory.

## 🪐 Open Source Software Statement

Please refer to the [document](https://easyclimate-backend.readthedocs.io/en/latest/src/softlist.html).
