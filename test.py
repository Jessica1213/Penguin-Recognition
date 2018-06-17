#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 20:21:42 2018

@author: jessica
"""

from PIL import Image
im = Image.open("../CNNdataset/TRUE/T00021.jpg")
im.show()
print(im.size)
im = Image.open("../CNNdataset/TRUE/T00022.jpg")
im.show()
print(im.size)