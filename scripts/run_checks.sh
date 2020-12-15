#!/bin/bash

set -o xtrace

python setup.py check --strict --metadata --restructuredtext
check-manifest
flake8 src
isort --verbose --check-only --diff --filter-files src

#sphinx-build -b doctest docs dist/docs
sphinx-build -b html docs dist/docs
sphinx-build -b linkcheck docs dist/docs

python setup.py clean --all bdist_wheel sdist
# skip: bdist
twine check dist/*.{whl,gz}
# twine upload --skip-existing dist/*.{whl,gz}
