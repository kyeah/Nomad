#!/bin/sh

python track_planar.py \
    --merge-contours \
    --costfunc rect \
    --stream \
    --all \
    --tracker naive \
    --object content/cello_and_stand.obj \
    --corners content/blankpaper.mp4 \
    "$@"

