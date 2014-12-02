#!/usr/bin/env python

import math


def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def avg(gen):
    total = 0
    n = 0
    for i in gen:
        total += i
        n += 1
    return float(total) / float(n)
