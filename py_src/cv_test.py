import platform
import sys
import os
import math

import numpy as np
import cv2



arr = np.array([[20.0, 15.0, 37.0], [47.0, 13.0, np.nan]]).astype(np.float32)


avg = cv2.mean(arr)

print(avg[0])

avg2 = np.nanmean(arr)

print(avg2)

bp = 1
