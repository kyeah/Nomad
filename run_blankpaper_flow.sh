#!/bin/sh

python track_planar.py \
    --costfunc combined2 \
    --stream \
    --tracker pointFlow \
    --object content/cello_and_stand.obj \
    --corners content/blankpaper.mp4 \
    "$@"

