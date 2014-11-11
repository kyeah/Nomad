#!/usr/bin/env python

import motion_tween as mt

def main():
    
    tracker = mt.VideoTracker(videoFilename, overlayFilename, overlayBounds)
    for i, frame in enumerate(tracker.frames()):

        # write frame to image files or new video
        pass

if __name__ == '__main__':
    main()