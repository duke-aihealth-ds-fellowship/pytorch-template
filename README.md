# pytorch-template
A template for PyTorch projects.

### Goal
This project is a template to jump start PyTorch projects. It is designed to be simple and hackable. It is not meant to be a one-size-fits-all solution, but a starting point that can be modified as needed. It is structured as a Python package (see this [tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)) to ease dissemination and reuse. When beneficial, it follows best practices described [here](https://github.com/Elliot-D-Hill/best-practice). 

### Installation
```bash
git clone https://github.com/Elliot-D-Hill/pytorch-template.git
```
After cloning the repo, give the package a new name (currently set to "example" in this repo) by modifying the `name` variable in `pyproject.toml` and in the package directory name `src/example`. For example, if you wanted to changed the name to `mypackage`, the package directory would become `src/mypackage`. Next, install the package with:
```bash
pip install -e path/to/cloned/repo
```
Now you can easily import modules, functions, and classes from your package with. For example:
```python
from mypackage.model import MyModel
```

### Usage
Once installed, you can run the package with:
```bash
python3 -m mypackage
```

### Structure
All arguments that affect program behavior are consolidated in `config.toml`. This allows for easy modification of program behavior without changing the source code. The `config.toml` file is parsed and validated with Pydantic and can be accessed as a Python object in the scripts. The file `__main__.py` is the entry point for the package where all high-level control flow is defined.

### Implemented components
- Configuration with Pydantic
- Custom PyTorch dataset
- Dataloaders for train, validation, and test splits
- A training loop
- Model checkpointing
- Hyperparameter tuning with Optuna
- Model evaluation with TorchMetrics

### Toy dataset examples
- A custom PyTorch dataset for sequence data e.g. word embeddings
- More to come...

### Design philosophy
- Simple is better than clever
- Explicit is better than implicit
- Modular is better than monolithic
- Practicality over purity