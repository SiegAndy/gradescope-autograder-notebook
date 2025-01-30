#!/usr/bin/env bash

rm -f P1-train.zip

zip -r P1-train.zip \
    train/msmarco.qrels train/msmarco.queries \
    train/*.trecrun \
    output/*.expeval
