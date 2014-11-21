#!/bin/sh

python track_planar.py \
    --merge-contours \
    --costfunc combined2 \
    --stream \
    --object content/humanoid_tri.obj \
    --corners content/music.mp4 content/music_training.png \
    "$@"

