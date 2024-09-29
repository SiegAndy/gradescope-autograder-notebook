#!/usr/bin/env bash

zip -r autograder.zip setup.sh run_autograder run_tests.py solution.py requirements.txt tests/*.py data/P1-train.gz data/stopwords.txt
