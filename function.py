# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:22:41 2019

@author: kk
"""

import numpy as np
import copy as cp

def move(board,x,y,symbol):
    if board[x][y] != "_": return False
    new_state = cp.deepcopy(board)
    new_state[x][y] = symbol
    return new_state

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def convert(state):
    state1=np.reshape(state,[1,state.size])
    string = ""
    for x in state1[0]:
        string=string+x
    return string
def convert_input(state):    
    state_input = np.full([1,state.size],0)
    s = np.reshape(state,[1,state.size])
    for i in range(state.size):
        if s[0][i] == "x":
            state_input[0][i] = -1
        elif s[0][i] == "o":
            state_input[0][i] = 1
        else:
            state_input[0][i] = 0
    return state_input[0]
def score(board,symbol,other_symbol):
    for symbol, point in zip(other_symbol+symbol, [-1,1]):
        
            if (board == symbol).all(axis=1).any(): return point
            if (board == symbol).all(axis=0).any(): return point
            if (np.diagonal(board) == symbol).all(): return point
            if (np.diagonal(np.rot90(board)) == symbol).all(): return point
    return 0           
    

def get_choice(choice,n):
        
        coordinate={}
        for j in range(n*n):       
            coordinate[j] = (j//n,j%n)
        return coordinate.get(choice)