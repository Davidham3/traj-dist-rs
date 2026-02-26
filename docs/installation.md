# Installation

You can install `traj-dist-rs` using pip:

## From PyPI

```bash
pip install traj-dist-rs
```

## From Source

uv and rustup are required.

```bash
git clone https://github.com/Davidham3/traj-dist-rs.git
cd traj-dist-rs
uv pip install .
```

## Development Setup

To set up a development environment, uv and rustup are required:

```bash
git clone https://github.com/Davidham3/traj-dist-rs.git
cd traj-dist-rs
uv sync --dev --extra test
```

This will build the Rust extension and install the package in development mode.
