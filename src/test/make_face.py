import dlib
import cv2

path = 'temp.jpg'

img = cv2.imread(path)
h, w = img.shape[:2]
detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)
if dets is None or len(dets) != 1:
    print("unsuitable")
    exit()
d = dets[0]
if d.left() < 0 or d.top() < 0 or d.right() > w or d.bottom() > h:
    print("unsuitable")
    exit()
face = img[d.top():d.bottom(), d.left():d.right()]
face = cv2.resize(face, (96, 96))
cv2.imwrite('test.jpg', face)
