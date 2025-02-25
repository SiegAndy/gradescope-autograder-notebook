#!/usr/bin/env bash

rm -f autograder.zip

cp data/2025-Spring-P2/run_autograder .

zip -r autograder.zip \
    setup.sh run_autograder run_tests.py requirements.txt tests/*.py tests/PA2/*.py \
    data/P2-train.gz data/PandP.gz data/webpage.gz

rm run_autograder
