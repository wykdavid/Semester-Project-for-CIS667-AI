# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:58:14 2019

@author: kk
"""

import tensorflow as tf

from expectminimaxAgent import ExpectMinMaxAgent
import function
from board import board
from randomAgent import baselineOpponent
import matplotlib.pyplot as plt





def Play(b,player,agent,game_number): 
    agentwin=0
    agentloss=0
    draw=0
    board = b.state
    for i in range(game_number):

         while True:
             
            
            board=player.move(board,agent.other_symbol)
            
            
            s=function.score(board,agent.symbol,agent.other_symbol)
            if s==-1:
                agentloss+=1
                agent.fallback(board,s)
                player.fallback(board,s)
                
                print("Player win"+str(agentloss))
                

                board = b.newgame()
                break
            
            if s==0 and "_" not in board:
                agent.fallback(board,s)
                player.fallback(board,s)
                
                draw+=1
                print("Draw"+str(draw))
                board = b.newgame()
                break
            
            board = agent.move(board,agent.symbol)
            
            s=function.score(board,agent.symbol,agent.other_symbol)
            
            
            

            if s==1:
                agentwin+=1
                agent.fallback(board,s)
                player.fallback(board,s)
                
                print("Agent win"+str(agentwin))
                board = b.newgame()
                break
            
            if s==0 and "_" not in board:
                draw+=1
                agent.fallback(board,s)
                player.fallback(board,s)
                
                print("Draw"+str(draw))
                board = b.newgame()
                break
    return agentwin,agentloss,draw

def eva(board,player1,player2,gamesize,iteration):
    p1w = []
    p2w = []
    draw = []
    game_count = []
    count = 0
    node = []
    for i in range(iteration):
        p1,p2,d = Play(board,player1,player2,gamesize)
        p1w.append(p1)
        p2w.append(p2)
        draw.append(d)
        count += 1
        game_count.append(count)
        node.append(player2.getnodes())
        player2.clearmemory()
        print("battle"+str(count))
    return game_count,p1w,p2w,draw,node
    
if __name__ == "__main__":
    tf.reset_default_graph()  
    problem_size = 3
    game_number = 100
    iteration = 100
    depth_limit = 7
    b = board(problem_size)
    randplayer1 = baselineOpponent("x","o")
    randplayer2 = baselineOpponent("o","x")
    mmAgent = ExpectMinMaxAgent("o","x",problem_size,0.95,0.01,depth_limit)
    c,x,y,z,n=eva(b,randplayer1,mmAgent,game_number,iteration)
    plt.figure()
    p = plt.plot(c, z, 'r-', c, x, 'g-', c, y, 'b-')
    plt.figure()
    p2 = plt.plot(c,n,"r-")
    
    
