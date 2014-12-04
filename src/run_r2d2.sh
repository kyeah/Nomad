#!/bin/sh

# Paint on R2D2! Draw with the mouse, then press 'p' to start tracking the painted object. Press 'l' to play the full video.

python track_planar.py \
    --stream \
    --no-viz \
    --stall \
    content/r2d2.mp4
    "$@"
