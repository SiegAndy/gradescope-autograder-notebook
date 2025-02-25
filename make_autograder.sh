#!/usr/bin/env bash

rm -f autograder.zip

cp data/2025-Spring-P1/run_autograder .

zip -r autograder.zip \
    setup.sh run_autograder run_tests.py requirements.txt tests/*.py tests/PA1/*.py \
    data/P1-train.zip data/2025-Spring-P1/data/train/*.* data/2025-Spring-P1/data/test/*.*

rm run_autograder
