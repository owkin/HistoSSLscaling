#!/bin/bash

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Necessary to propagate error codes.
set -e
set -o pipefail

# Black.
echo "Black formatting..."
black --line-length=79 --check --diff rl_benchmarks/ tools/ tests/

# Flake8.
echo -e "\nFlake8 formatting..."
echo -e "Number of errors:"
flake8 --count rl_benchmarks/ tools/ tests/

# Pylint.
echo -e "\nPylint formatting..."
export score=$(pylint --rcfile=pylintrc rl_benchmarks/ tools/ tests/ | sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p')

echo "Code has been rated at ${score}/10."

