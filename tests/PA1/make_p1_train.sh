#!/usr/bin/env bash

echo "creating P1-train.zip"
rm -f P1-train.zip

zip -r P1-train.zip \
    train/msmarco.qrels train/msmarco.queries \
    train/*.trecrun

echo "creating P1-train-output.zip"
rm -f P1-train-output.zip

zip -r P1-train-output.zip \
    output/*.expeval
