#!/usr/bin/env python
import freenect
import numpy as np
import matplotlib.pyplot as plt
import time


for i in range(100):
    # Get a fresh frame
    depth,_ = freenect.sync_get_depth()
    rgb,_ = freenect.sync_get_video()

    print(np.array(rgb).shape)
    # Build a two panel color image
    d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
    da = np.hstack((d3,rgb))
    # Simple Downsample
    print('working')
    time.sleep(1)



print('Running')
