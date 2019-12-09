# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:42:32 2019

@author: kk
"""

import numpy as np
import copy as cp
import random
import tensorflow as tf
from Network import DQNAgent 
#from board import board
import function


# =============================================================================
# def move(state,x,y,symbol):
#     if board[x][y] != "_": return False
#     new_state = cp.deepcopy(state)
#     new_state[x][y] = symbol
#     return new_state
# =============================================================================

# =============================================================================
# def softmax(x):
#     x = x - np.max(x)
#     exp_x = np.exp(x)
#     softmax_x = exp_x / np.sum(exp_x)
#     return softmax_x
# =============================================================================

# =============================================================================
# def convert(state):
#     state1=np.reshape(state,[1,state.size])
#     string = ""
#     for x in state1[0]:
#         string=string+x
#             
#     return string
# =============================================================================

# =============================================================================
# def get_choice(choice,n):
#         
#         coordinate={}
#         for j in range(n*n):       
#             coordinate[j] = (j//n,j%n)
#         return coordinate.get(choice)
# =============================================================================
    

class ExpectMinMaxAgent:
    
    def __init__(self,symbol,other_symbol,board_size,gamma,lr):
        self.symbol = symbol
        self.other_symbol = other_symbol
        self.board_size = board_size
        self.memory = {}
        self.depth = 0
        self.agent = DQNAgent("dqn",gamma,lr,True,symbol,board_size)
        super().__init__()
    

    def available_area(self,board,action):
        neighbor_count = 5
        
        neighbour = [action,action-self.board_size,action+self.board_size,action-1,action+1]
        if action-self.board_size < 0:
            neighbor_count -= 1
            neighbour.remove(action-self.board_size)
        
        if action+self.board_size > (self.board_size*self.board_size)-1:
            neighbor_count -= 1
            neighbour.remove(action+self.board_size)
        
        if action%self.board_size == 0:
            neighbor_count -= 1
            neighbour.remove(action-1)
        
        if action%self.board_size == self.board_size-1:
            neighbor_count -= 1
            neighbour.remove(action+1)
        

        actual_neighbour=[]
        for n in neighbour:            
            coordinate = function.get_choice(n,self.board_size)
            
            if board[coordinate[0]][coordinate[1]] != "_":
                neighbor_count -= 1
                
            else:
                actual_neighbour.append(n)

        return neighbor_count,actual_neighbour
    def get_max(self,board):
        state1=np.reshape(board,[1,self.board_size*self.board_size])
        depth_count = 0
        for i in state1[0]:
            if i != "_":
                depth_count += 1
# =============================================================================
#         print(board)
#         print(depth_count)
# =============================================================================
# =============================================================================
#         if depth_count-1 >=self.depth:
#             self.depth = depth_count-1
# =============================================================================
        hash_state = function.convert(board)
        if hash_state in self.memory:
                return self.memory[hash_state]
        if depth_count-1 < 3:
            
            
            s = function.score(board,self.symbol,self.other_symbol)
            next_action = -1
            if s != 0 :
                next_action = -1
                self.memory[hash_state] = (s,next_action)
            elif s==0 and "_" not in board:
                next_action = -1
                self.memory[hash_state] = (s,next_action)
            else:
                max_value = 0
                
                for i in range(self.board_size*self.board_size):
                    coordinate = function.get_choice(i,self.board_size)                
                    if board[coordinate[0]][coordinate[1]] == "_":
                        theta,possible_choice = self.available_area(board,i)
                        expect_value = 0.
                        for action in possible_choice:
                            children = cp.deepcopy(board)
                            sub_coordinate = function.get_choice(action,self.board_size)                        
                            children[sub_coordinate[0]][sub_coordinate[1]]  = self.symbol
                            value,_ = self.get_min(children)
                            expect_value += value/theta
                        
                        if expect_value > max_value or next_action == -1:
                            max_value = expect_value
                            #print(expect_value)
                            next_action  = i
                            
                            if max_value == 1:
                                self.memory[hash_state] = (max_value,next_action)
                                return max_value,next_action
                        self.memory[hash_state] = (max_value,next_action)
                
            return self.memory[hash_state]
        
        else:
  
            state = cp.deepcopy(board)
            s = function.score(state,self.symbol,self.other_symbol)
            next_action = -1
            if s != 0 :
                next_action = -1
                self.memory[hash_state] = (s,next_action)
            elif s==0 and "_" not in board:
                next_action = -1
                self.memory[hash_state] = (s,next_action)
          
            while True:
                
                state = self.agent.move(state)
                s = function.score(state,self.symbol,self.other_symbol)
                if s == 1:
                    self.agent.fallback(state,s)
                    break
                if s == 0 and "_" not in state:
                    self.agent.fallback(state,s)
                    break
                if "_" in state:
                    playerx,playery=random.randint(0,self.board_size-1),random.randint(0,self.board_size-1)
                    statep = function.move(state,playerx,playery,self.other_symbol)
                    while statep is False :
                        playerx,playery=random.randint(0,self.board_size-1),random.randint(0,self.board_size-1)
                        statep = function.move(state,playerx,playery,self.other_symbol)
                    state=statep
                s = function.score(state,self.symbol,self.other_symbol)
                if s == -1:
                    self.agent.fallback(state,s)
                    break
                if s == 0 and "_" not in state:
                    self.agent.fallback(state,s)
                    break
            action = self.agent.history[0]
            softmax_value = function.softmax(self.agent.value[0])
            max_value = np.max(softmax_value)
            
            self.memory[hash_state] = (max_value,action)
            self.agent.clean()
        return self.memory[hash_state]
            #需要softmax 最大为1 返回选取位置和数值
    def get_min(self,board):
        state1=np.reshape(board,[1,self.board_size*self.board_size])
        depth_count = 0
        for i in state1[0]:
            if i == "_":
                depth_count += 1
        if depth_count-1 >=self.depth:
            self.depth = depth_count-1
        
        hash_state = function.convert(board)
        if hash_state in self.memory:
            return self.memory[hash_state]
        #self.depth += 1
        s = function.score(board,self.symbol,self.other_symbol)
        next_action = -1
        if s != 0 :
            next_action = -1
            self.memory[hash_state] = (s,next_action)
            
        elif s==0 and "_" not in board:
            next_action = -1
            self.memory[hash_state] = (s,next_action)
            
        else:
            min_value = 0
            
            for i in range(self.board_size*self.board_size):
                coordinate = function.get_choice(i,self.board_size)
                children = cp.deepcopy(board)
                if board[coordinate[0]][coordinate[1]] == "_":
                    children[coordinate[0]][coordinate[1]]  = self.other_symbol
                    value,_ = self.get_max(children)
                    if value < min_value or next_action == -1:
                        min_value = value
                        next_action = i
                        
                        if min_value == -1:
                            self.memory[hash_state] = (min_value,next_action)
                            return self.memory[hash_state]
                        
                    self.memory[hash_state] = (min_value,next_action)
             
        return self.memory[hash_state]
          
              
              
    def move(self, board, symbol):

        score , next_action = self.get_max(board)
# =============================================================================
#         print(board)
#         print("choice "+str(next_action))
# =============================================================================
        count,possible_action = self.available_area(board,next_action)
        random_action = random.randint(0,count-1)
        actual_action = possible_action[random_action]

        coordinate = function.get_choice(actual_action,self.board_size)
        
        board[coordinate[0]][coordinate[1]] = self.symbol
        
        return board
    def fallback(self,board,s):
        #print(self.depth)
        pass