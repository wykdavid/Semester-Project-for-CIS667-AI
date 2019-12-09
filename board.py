# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:17:13 2019

@author: kk
"""

import numpy as np

class board:
    def __init__(self,size):
        self.mysize = size
        self.state = np.full([size,size],"_")
    def newgame(self):
        self.state = np.full([self.mysize,self.mysize],"_")
        return  np.full([self.mysize,self.mysize],"_")
    def getsize(self):
        return self.mysize
       