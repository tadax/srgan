import os
import glob
import cv2
import numpy as np

def load(dir_):
    x = []; t = []
    paths = glob.glob(os.path.join(dir_, '*'))
    for path in paths:
        bgr_img = cv2.imread(path)
        rbg_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img = np.array(rbg_img) / 127.5 - 1
        x.append(img)
    return np.array(x)

