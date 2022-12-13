import logging

import numpy as np
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def autoContrast(img, _):   # 0
    return PIL.ImageOps.autocontrast(img)


def brightness(img, b):     # 1         
    return PIL.ImageEnhance.Brightness(img).enhance(b)


def color(img, c):          # 2
    return PIL.ImageEnhance.Color(img).enhance(c)


def contrast(img, c):       # 3
    return PIL.ImageEnhance.Contrast(img).enhance(c)


def equalize(img, _):       # 4
    return PIL.ImageOps.equalize(img)


def identity(img, _):       # 5
    return img


def posterize(img, b):      # 6
    return PIL.ImageOps.posterize(img, int(b))


def rotate(img, theta):     # 7
    return img.rotate(theta)


def sharpness(img, s):      # 8
    return PIL.ImageEnhance.Sharpness(img).enhance(s)


def shear_x(img, r):        # 9
    return img.transform(img.size, Image.AFFINE, (1, r, 0, 0, 1, 0))


def shear_y(img, r):        # 10
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, r, 1, 0))


def solarize(img, t):       # 11
    return PIL.ImageOps.solarize(img, t)


def translate_x(img, l):    # 12
    return img.transform(img.size, Image.AFFINE, (1, 0, l, 0, 1, 0))


def translate_y(img, l):    # 13
    return img.transform(img.size, Image.AFFINE, (1, 0, 1, 0, 1, l))


def rand_augment_transformations():
    transformations = [
        (autoContrast, None, None),
        (brightness, 0.05, 0.95),
        (color, 0.05, 0.95),
        (contrast, 0.05, 0.95),
        (equalize, None, None),
        (identity, None, None),
        (posterize, 4, 8),
        (rotate, -30, 30),
        (sharpness, 0.05, 0.95),
        (shear_x, -0.3, 0.3),
        (shear_y, -0.3, 0.3),
        (solarize, 0, 1),
        (translate_x, -0.3, 0.3),
        (translate_y, -0.3, 0.3)
        ]
    return transformations

class SoftAugment():
    def __init__(self):
        pass

    def __call__(self, img):
        return img

class RandAugment():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.transformations = rand_augment_transformations()

    def __call__(self, img):
        img = Image.fromarray(np.uint8(np.transpose(img, (2, 1, 0))*255), 'RGB')
        selected_index = np.random.choice(len(self.transformations), self.n, replace=False)
        selected_transformations = np.array(self.transformations)[selected_index]
        logger.debug(selected_index)
        for transformation, min, max in selected_transformations:
            if np.random.rand() > 0.5:
                amplitude = np.random.randint(0, self.m)
                #parameter = np.random.rand() * (max - min) + min if min is not None else 0 
                parameter = float(amplitude) * (max - min) / 10 + min if min is not None else 0
                img = transformation(img, parameter)
        
        return np.transpose(img, (2, 1, 0))/255