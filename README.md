# pytorch-template

A template for PyTorch projects.

## Goal

This project is a template to jump-start PyTorch projects. It is designed to be simple and hackable. It is not meant to be a one-size-fits-all solution, but a starting point that can be modified as needed. It is structured as a Python package (see this [tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)) to ease dissemination and reuse. When beneficial, it follows best practices described [here](https://github.com/Elliot-D-Hill/best-practice).

## Installation

Fork or clone the repo from the web or command line (you will need the GitHub CLI installed to use `gh`):

```bash
gh repo fork https://github.com/duke-aihealth-ds-fellowship/pytorch-template.git --clone
```

 After forking the repo, give the package a new name (currently set to "template" in this repo) by modifying the `name` variable in `pyproject.toml` and in the package directory name `src/template`. For example, if you wanted to changed the name to `mypackage`, the package directory would become `src/mypackage`. Next, install the package with:

```bash
pip install -e path/to/repo
```

The argument argument `-e` stands for 'editable' mode. This setting makes it so that changes made to the package are reflected in the code behavior in real-time so that you don't have to keep reinstalling the package when you modify it.

Now you can easily import modules, functions, and classes from your package. For example:

```python
from mypackage.model import MyModel
```

## Usage

Once installed, you can run the package with:

```bash
python -m mypackage
```

## Structure

All arguments that affect program behavior are consolidated in `config.toml`. This allows for easy modification of program behavior without changing the source code. The `config.toml` file is parsed and validated with Pydantic and can be accessed as a Python object in the scripts. The file `__main__.py` is the entry point for the package where all high-level control flow is defined.

## Features

- Configuration validation with toml + Pydantic
- Tokenization
- Dataloaders for train, validation, and test splits
- A trainer with training and evaluation loop
- Hyperparameter tuning with Optuna
- Checkpointing
- Learning rate scheduling
- Model evaluation with TorchMetrics
- Feature importance with Captum
- Metric and feature importance plots with Seaborn/Matplotlib

## Toy dataset examples

- AG news text classification dataset
- More to come...

## Design philosophy

We prefer

- Simple over clever
- Explicit over implicit
- Modular over monolithic
- Practicality over purity
- Readability over conciseness

## Style guide

- Default `ruff` formatting
- Type hints are required (though not enforced)
- Names should be descriptive, concise, and consistent
- Variable names are snake_case
- Class names are CamelCase
- Constants are UPPER_CASE

### TODO

- Logging (e.g. Tensorboard)
- Distributed training
- Tokenization
- More toy datasets modalities, e.g., images, tabular, etc.
- Save configuration file after the final model is trained
- Automate path construction
- Tests
