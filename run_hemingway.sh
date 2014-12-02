#!/bin/sh

python track_planar.py \
    --merge-contours \
    --costfunc rect \
    --stream \
    --all \
    --object content/cello_and_stand.obj \
    --corners content/hemingway.mp4 \
    "$@"

