# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:09:26 2019

@author: kk
"""
import function
import random
import math
class baselineOpponent:
    def __init__(self,symbol,other_symbol):
        self.symbol = symbol
        self.other_symbol = other_symbol
    def move(self,board,symbol):
        #if "_" in board:
        playerx,playery=random.randint(0,int(math.sqrt(board.size)-1)),random.randint(0,int(math.sqrt(board.size)-1))
        statep = function.move(board,playerx,playery,symbol)
        while statep is False:
            playerx,playery=random.randint(0,int(math.sqrt(board.size)-1)),random.randint(0,int(math.sqrt(board.size)-1))
            statep = function.move(board,playerx,playery,symbol)
        return statep
    def fallback(self,board,s):
        pass
    def clearmemory(self):
        pass
    def getnodes(self):
        pass