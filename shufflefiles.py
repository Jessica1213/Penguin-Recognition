#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 01:25:33 2018

@author: jessica
"""

import os #os module imported here
from random import shuffle

truedir = '../CNNdataset/TRUE'
falsedir = '../CNNdataset/FALSE'
truefiles = []
falsefiles = []

for file in os.listdir(truedir):
    try:
        truefiles.append(file)
    except Exception as e:        
        raise e
        
for file in os.listdir(falsedir):
    try:
        falsefiles.append(file)
    except Exception as e:
        raise e


shuffle(truefiles)
shuffle(falsefiles)

truetrainsize = int(len(truefiles)*0.9)
falsetrainsize = int(len(falsefiles)*0.9)
file = open('truetrain.txt', "w")
for name in truefiles[:truetrainsize]:
    file.write(name+"\n")
file.close()
file = open('truetest.txt', "w")
for name in truefiles[truetrainsize:]:
    file.write(name+"\n")
file.close()    
file = open('falsetrain.txt', "w")
for name in falsefiles[:falsetrainsize]:
    file.write(name+"\n")
file.close()    
file = open('falsetest.txt', "w")
for name in falsefiles[falsetrainsize:]:
    file.write(name+"\n")
file.close()