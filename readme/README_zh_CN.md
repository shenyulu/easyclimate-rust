<img src="https://github.com/shenyulu/easyclimate-rust/blob/main/docs/logo/easyclimate_rust_logo_mini.png?raw=true" alt="easyclimate-rust">

<h2 align="center">easyclimate 的 Rust 后端</h2>

![PyPI - 版本](https://img.shields.io/pypi/v/easyclimate-backend)
![PyPI - Python版本](https://img.shields.io/pypi/pyversions/easyclimate-backend)
![PyPI - 下载量](https://img.shields.io/pypi/dm/easyclimate-backend)
[![文档状态](https://readthedocs.org/projects/easyclimate-backend/badge/?version=latest)](https://easyclimate-backend.readthedocs.io/en/latest/?badge=latest)

<div align="center">
<center><a href = "../README.md">English</a> / 简体中文 / <a href = "README_ja_JP.md">日本語</a></center>
</div>

## 🤗 easyclimate-rust 是什么？

**easyclimate-rust** 是一个使用 Rust 编写的高性能后端库，旨在为
[easyclimate](https://github.com/shenyulu/easyclimate)
提供计算密集型任务的支持。

通过充分利用 Rust 的 **高性能**、**内存安全性** 以及 **零成本抽象**，
easyclimate-rust 使 Python 前端能够在保持接口简洁、易用的同时，
在处理大规模气候数据和复杂诊断计算时依然具备良好的可扩展性和效率。

> 🚨 **项目状态：积极开发中** 🚨  
>
> 本项目仍处于快速迭代阶段。
> API（包括函数、类及接口）**尚未稳定**，未来版本可能在不保证向后兼容的情况下发生变更。
> 请谨慎用于生产环境。

## 😯 安装方式

可以通过 Python 包管理器
[pip](https://pip.pypa.io/en/stable/getting-started/)
安装 `easyclimate-rust`：

```bash
pip install easyclimate-rust
````

## ✨ 环境要求

* **Python** ≥ 3.10
* **NumPy** ≥ 1.24.3
  *(仅在使用预编译 wheel 包运行时需要)*

## 🔧 构建说明

### Windows

1. 安装 Rust
   👉 [在 Windows 上针对 Rust 设置开发环境](https://learn.microsoft.com/zh-cn/windows/dev-environment/rust/setup)

2. 安装 `uv`：

```powershell
winget install uv
```

3. 在项目根目录运行构建脚本：

```powershell
.\scripts\build_manywindows_wheel.ps1
```

4. 生成的 wheel 文件将位于 `dist/` 目录中。

### Linux

1. 在系统中安装 Docker。
2. 在 Linux 主机上运行构建脚本：

```bash
./scripts/build_manylinux_wheel.sh
```

生成的 wheel 文件同样位于 `dist/` 目录中。

## 🪐 开源软件声明

请参阅[说明文档](https://easyclimate-backend.readthedocs.io/en/latest/src/softlist.html)。
