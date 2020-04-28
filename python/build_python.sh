#!/usr/bin/env bash

echo "Running python tests"
pytest --profile-svg

# python linting
echo "Running python linter"
pylint -j 8 --rcfile=.pylintrc motifcluster

# python rtfd
echo "Building python docs"
cd doc/
make html latex latexpdf
cp _build/latex/motifcluster.pdf .

# python distribution packages
cd ..
echo "Building python distribution packages"
python setup.py sdist bdist_wheel

# python upload to PyPI
#python -m twine upload dist/*
### username: __token__
### password: <PyPI API token>
