#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 20:21:42 2018

@author: jessica
"""

from PIL import Image
image = Image.open("../CNNdataset/TRUE/T00035.jpg")
try:    
    exif=dict(image._getexif().items())
    orientation = 274
    if   exif[orientation] == 3 : 
        image=image.rotate(180, expand=True)
    elif exif[orientation] == 6 : 
        image=image.rotate(270, expand=True)
    elif exif[orientation] == 8 : 
        image=image.rotate(90, expand=True)
except:
    pass
image.show()
print(image.size)

