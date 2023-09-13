#!/bin/bash

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

taskset -c 0-11 pytest -v -m 'not test_raw_data_loading' --cov=rl_benchmarks --cov=tools --cov-report=xml --durations=0 tests/
