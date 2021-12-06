import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def custom_resize(img):
    # interp = sample_discrete('bilinear')
    return torchvision.transforms.functional.resize(img, 256, interpolation='bilinear')
