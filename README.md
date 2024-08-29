# pytorch-template
A simple, hackable template for PyTorch projects.

### Goal
This project is a template to jump start PyTorch projects. It is structured as a Python package (see this [tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)) to ease dissemination and reuse.

### Installation
```bash
git clone path/to/repo
pip install -e path/to/repo
```

### Usage
Give the package a new name (set to "example" by default) by modifying it in pyproject.toml and the package directory in `src`. After which you can run the package with:
```bash
python3 -m your_package_name
```

### Implements:
- Configuration with Pydantic
- Dataloaders for train, validation, and test splits
- A barebones training loop
- Model checkpointing
- Hyperparameter tuning with Optuna
- Model evaluation with TorchMetrics

### Toy dataset examples
- A custom PyTorch dataset for sequence data e.g. word embeddings
- More to come...
