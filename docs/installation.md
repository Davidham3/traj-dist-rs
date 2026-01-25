# Installation

You can install `traj-dist-rs` using pip:

## From PyPI

```bash
pip install traj-dist-rs
```

## From Source

```bash
git clone https://github.com/your-repo/traj-dist-rs.git
cd traj-dist-rs
pip install maturin
maturin build --release
pip install dist/*.whl
```

## Development Setup

To set up a development environment:

```bash
git clone https://github.com/your-repo/traj-dist-rs.git
cd traj-dist-rs
pip install maturin
maturin develop
```

This will build the Rust extension and install the package in development mode.