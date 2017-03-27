import os
import glob
import cv2 
import dlib
import shutil
import random
import numpy as np
import scipy.misc

temp_dir = './temp'
train_dir = './train'
test_dir = './test'

class Data:
    def __init__(self, img_dim):
        self.ratio = 0.05
        self.img_dim = img_dim
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, img):
        dets = self.detector(img, 1)
        if dets is None or len(dets) > 1:
            return None
        h_img, w_img = img.shape[:2]
        det = dets[0]
        if (int(det.left()) < 0 or int(det.top()) < 0 or int(det.right()) > w_img or int(det.bottom()) > h_img:
            continue
        cropped_img = img[det.top():det.bottom(), det.left():det.right()]
        return cv2.resize(cropped_img, (self.dim, self.dim))

    def preprocess(self):
        persons = glob.glob('lfw/*') 
        for l, person in enumerate(persons):
            l_ = "{0:09d}".format(l)
            paths = glob.glob(os.path.join(person, '*'))
            for l, path in enumerate(glob.glob(os.path.join(person, '*'))):
                i_ = "{0:04d}".format(i) 
                bgr_img = cv2.imread(path) 
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face = self.detect_faces(rgb_img)
                if face is not None:
                    name = "{}_{}.jpg".format(l_, i_)
                    path = os.path.join(temp_dir, name)
                    scipy.misc.imsave(path, np.uint8(face))

    def split_into_train_and_test(self):
        images = np.array([os.path.basename(p) for p in glob.glob(os.path.join(temp_dir, '*'))])
        perm = np.random.permutation(len(images))
        images = images[perm]
        data_list = {}
        for image in images:
            person = image.split('_')[0]
            if person in data_list:
                data_list[person].append(image)
            else:
                data_list[person] = [image]
        n_classes = 0
        for person, data in data_list.items():
            n = int(len(data) * (1 - self.ratio))
            if n < 1:
                continue
            else:
                n_classes += 1
            print("{}: train = {}, test = {}".format(person, len(data)-n, n))
            data.sort()
            for l in data[:n]:
                src = os.path.join(temp_dir, l)
                dst = os.path.join(test_dir, l)
                shutil.move(src, dst)
            for l in data[n:]:
                src = os.path.join(temp_dir, l)
                dst = os.path.join(train_dir, l)
                shutil.move(src, dst)
        shutil.rmtree(temp_dir)
        print("class num: {}".format(n_classes))


if __name__ == "__main__":
    if input("Are you sure to make data newly? (y/n)\n") != 'y':
        exit()
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    img_dim = 96
    data = Data(img_dim=img_dim)
    data.preprocess()

