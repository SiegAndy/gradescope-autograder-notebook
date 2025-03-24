#!/usr/bin/env bash

rm -f autograder.zip
rm -f autograder-2025-p3.zip

cp data/2025-Spring-P3/run_autograder .

zip -r autograder-2025-p3.zip \
    setup.sh run_autograder run_tests.py requirements.txt tests/*.py tests/PA1/*.py \
    data/P1-train.zip data/2025-Spring-P3/data/train/*.* data/2025-Spring-P1/data/test/*.*

rm run_autograder
