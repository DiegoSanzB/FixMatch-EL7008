import logging

import numpy as np
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def autoContrast(img, _): 
    return PIL.ImageOps.autocontrast(img)


def brightness(img, b):    
    return PIL.ImageEnhance.Brightness(img).enhance(b)


def color(img, c):
    return PIL.ImageEnhance.Color(img).enhance(c)


def contrast(img, c):
    return PIL.ImageEnhance.Contrast(img).enhance(c)


def equalize(img, _):
    return PIL.ImageOps.equalize(img)


def identity(img, _):
    return img


def posterize(img, b):
    return PIL.ImageOps.posterize(img, int(b))


def rotate(img, theta): 
    return img.rotate(theta)


def sharpness(img, s): 
    return PIL.ImageEnhance.Sharpness(img).enhance(s)


def shear_x(img, r): 
    return img.transform(img.size, Image.AFFINE, (1, r, 0, 0, 1, 0))


def shear_y(img, r): 
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, r, 1, 0))


def solarize(img, t): 
    return PIL.ImageOps.solarize(img, t)


def translate_x(img, l): 
    return img.transform(img.size, Image.AFFINE, (1, 0, l, 0, 1, 0))


def translate_y(img, l):
    return img.transform(img.size, Image.AFFINE, (1, 0, 1, 0, 1, l))


def flip(img, _):
    return PIL.ImageOps.flip(img)


def mirror(img, _):
    return PIL.ImageOps.mirror(img)


class WaveDeformer:

    def transform(self, x, y):
        y = y + 10*np.sin(x/4)
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (*self.transform(x0, y0),
                *self.transform(x0, y1),
                *self.transform(x1, y1),
                *self.transform(x1, y0),
                )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = 20

        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))

        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]

        return [t for t in zip(target_grid, source_grid)]


def deform(img, _):
    return PIL.ImageOps.deform(img, deformer=WaveDeformer())


def rand_augment_transformations():
    transformations = [
        (autoContrast, None, None), # 0
        (brightness, 0.05, 0.95),   # 1
        (color, 0.05, 0.95),        # 2
        (contrast, 0.05, 0.95),     # 3
        (equalize, None, None),     # 4
        (identity, None, None),     # 5
        (posterize, 4, 8),          # 6
        (rotate, -30, 30),          # 7
        (sharpness, 0.05, 0.95),    # 8
        (shear_x, -0.3, 0.3),       # 9
        (shear_y, -0.3, 0.3),       # 10
        (solarize, 0, 1),           # 11
        (translate_x, -0.3, 0.3),   # 12
        (translate_y, -0.3, 0.3)    # 13
        ]
    return transformations

def my_augment_transformations():
    transformations = [
        (autoContrast, None, None), # 0
        (brightness, 0.05, 0.95),   # 1   
        (color, 0.05, 0.95),        # 2
        (contrast, 0.05, 0.95),     # 3
        (equalize, None, None),     # 4
        (identity, None, None),     # 5
        (rotate, -30, 30),          # 6
        (sharpness, 0.05, 0.95),    # 7
        (shear_x, -0.3, 0.3),       # 8
        (shear_y, -0.3, 0.3),       # 9
        (solarize, 0, 1),           # 10
        (translate_x, -0.3, 0.3),   # 11
        (translate_y, -0.3, 0.3),   # 12
        (flip, None, None),         # 13
        (mirror, None, None),       # 14
        (deform, None, None)        # 15
    ]
    return transformations

class SoftAugment():
    def __init__(self):
        pass

    def __call__(self, img):
        return img

class MyAugment():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.transformations = my_augment_transformations()

    def __call__(self, img):
        img = Image.fromarray(np.uint8(np.transpose(img, (2, 1, 0))*255), 'RGB')
        selected_index = np.random.choice(len(self.transformations), self.n, replace=False)
        selected_transformations = np.array(self.transformations)[selected_index]
        logger.debug(selected_index)
        for transformation, min, max in selected_transformations:
            if np.random.rand() > 0.5:
                amplitude = np.random.randint(0, self.m)
                parameter = float(amplitude) * (max - min) / 10 + min if min is not None else 0
                img = transformation(img, parameter)
        
        return np.transpose(img, (2, 1, 0))/255

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
                parameter = float(amplitude) * (max - min) / 10 + min if min is not None else 0
                img = transformation(img, parameter)
        
        return np.transpose(img, (2, 1, 0))/255