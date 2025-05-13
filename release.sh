#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
rm tests.log
rm dist* -r
rm -r .tox
rm -r .pytest_cache
rm -r build
rm -R lcbuilder-reqs
rm -R *egg-info
conda remove -n lcbuilder-reqs --all -y
set -e
tox -r > tests.log
tests_results=$(cat tests.log | grep "congratulations")
if ! [[ -z ${tests_results} ]]; then
  echo "Building"
  set +e
  rm dist* -r
  rm -r .tox
  rm -r .pytest_cache
  rm -r build
  rm -R lcbuilder-reqs
  conda remove -n lcbuilder-reqs --all -y
  set -e
  conda create -n lcbuilder-reqs python=3.11 -y
  conda activate lcbuilder-reqs
  python3 -m pip install pip -U
  python3 -m pip install setuptools -U
  python3 -m pip install Cython -U
  python3 -m pip install extension-helpers -U
  python3 -m pip install numpy==2.2.4
  python3 -m pip install pybind11
  python3 -m pip install .
  python3 -m pip list --format=freeze > requirements.txt
  conda deactivate
  git_tag=$1
  git pull
  sed -i '5s/.*/version = "'${git_tag}'"/' setup.py
  git add requirements.txt
  git add setup.py
  git commit -m "Preparing release ${git_tag}"
  git tag ${git_tag} -m "New release"
  git push && git push --tags
else
  echo "Failed tests"
fi
set +e
rm -R lcbuilder-reqs
rm dist* -r
rm -r .tox
rm -r .pytest_cache
rm -r build
rm -R *egg-info
conda remove -n lcbuilder-reqs --all -y
set -e
