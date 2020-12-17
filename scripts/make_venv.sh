#!/bin/bash

# create venv
python3 -m venv venv
source venv/bin/activate
# update pip etc.
pip install -U pip setuptools wheel

# install others (test)?
pip install cookiecutter tox

# install package
pip install -e .

# install dev tools (style, format, linting)
pip install -e .[dev] black pylint

# install pytest
pip install -r tests/requirements.txt

# install docs requirements
pip install -r docs/requirements.txt

# install pypi tools
pip install bumpversion twine
