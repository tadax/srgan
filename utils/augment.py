import numpy as np
import random
import cv2
   
class IMG:
    def __init__(self, normalized=True, flip=True, brightness=True, cropping=True, blur=True):
        self.normalized = normalized
        self.flip = flip
        self.brightness = brightness
        self.cropping = cropping
        self.blur = blur

    def augment(self, images):
        return np.array([self._augment(image) for image in images])

    def _augment(self, image):
        if self.normalized: image = self.denormalize(image)
        if self.cropping: image = self.random_crop_and_zoom(image)
        if self.flip: image = self.random_flip_left_right(image)
        if self.brightness: image = self.random_brightness(image)
        if self.blur: image = self.random_blur(image)
        if self.normalized: image = self.normalize(image)
        return image

    def normalize(self, image):
        return (image / 127.5 - 1)

    def denormalize(self, image):
        return ((image + 1) * 127.5).astype(np.uint8)

    def random_flip_left_right(self, image, seed=None):
        if random.randint(0, 1) == 1:
            image = cv2.flip(image, 1)
        return image

    def random_brightness(self, image, alpha=2.0, seed=None):
        gamma = np.random.rand() * alpha
        gf = [[255 * pow(i / 255, 1 / gamma)] for i in range(256)]
        table = np.reshape(gf, (256, -1))
        return cv2.LUT(image, table)

    def random_crop_and_zoom(self, image, alpha=0.1, seed=None):
        img_h, img_w = image.shape[:2]
        r = random.uniform(0, alpha)
        v1 = random.randint(0, int(r * img_h))
        v2 = random.randint(0, int(r * img_w))
        image = image[v1:(v1 + int((1 - r) * img_h)), v2:(v2 + int((1 - r) * img_w)), :]
        image = cv2.resize(image, (img_h, img_w))
        return image

    def random_blur(self, image, alpha=4, seed=None):
        f = random.randint(1, alpha)
        image = cv2.blur(image, (f, f))
        return image

