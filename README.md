# Minitorch

(with a better readme)

## Instructions for exercises

[Start here](https://minitorch.github.io/module0/module0/)

## Setup

```bash
git clone https://github.com/minitorch/minitorch  # or your/my fork, e.g. https://github.com/nrimsky/minitorch
cd minitorch
python3 -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements.extra.txt
pip install -Ue .
```

## Running tests for a task

```bash
pytest -m task-id # e.g. `pytest -m task0_1` for section with note "pass tests marked as task0_1"
```

_Remember to activate the virtual environment first (`source venv/bin/activate` or `venv\Scripts\activate` on Windows) before running the tests._
