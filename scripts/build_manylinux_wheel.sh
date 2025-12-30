#!/bin/bash

docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release --manylinux 2014 --find-interpreter
mkdir -p dist
cp ./target/wheels/*manylinux*.whl ./dist