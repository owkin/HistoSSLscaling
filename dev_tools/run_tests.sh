#!/bin/bash

taskset -c 0-11 pytest -v -m 'not test_raw_data_loading' --cov=rl_benchmarks --cov=tools --cov-report=xml --durations=0 tests/
