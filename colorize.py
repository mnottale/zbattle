#! /usr/bin/env python3


import cv2
import numpy as np
import sys

for f in sys.argv[1:]:
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    img[:,:,2] = 255
    cv2.imwrite(f[:-4] + '-3.png', img)
    img[:,:,1] = 255
    cv2.imwrite(f[:-4] + '-2.png', img)
    img[:,:,2] = 0
    img[:,:,0] = 255
    cv2.imwrite(f[:-4] + '-1.png', img)
    img[:,:,1] = 0
    img[:,:,2] = 255
    cv2.imwrite(f[:-4] + '-0.png', img)