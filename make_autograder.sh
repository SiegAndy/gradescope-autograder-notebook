#!/usr/bin/env bash

rm -f autograder.zip
rm -f autograder-2025-p2.zip

cp data/2025-Spring-P2/run_autograder .

zip -r autograder-2025-p2.zip \
    setup.sh run_autograder run_tests.py requirements.txt tests/*.py tests/PA2/*.py \
    data/P2-data.zip

rm run_autograder
