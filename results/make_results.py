import cv2
import scipy.misc

for i in range(1, 17):
    i = "{0:09d}".format(i)
    path = "../src/result/{}/000000020.jpg".format(i)
    x = cv2.imread(path)
    x = x[160:340, 70:585, :]
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) 
    path = "./{}.jpg".format(i)
    scipy.misc.imsave(path, x)

