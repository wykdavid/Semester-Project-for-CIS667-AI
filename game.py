# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:58:14 2019

@author: kk
"""

import tensorflow as tf
#from QL import QL
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
    #s = score(board)
# =============================================================================
         while True:
#             playerx,playery=random.randint(0,b.getsize()-1),random.randint(0,b.getsize()-1)
#             statep = function.move(board,playerx,playery,"x")
#             while statep is False:
#                 playerx,playery=random.randint(0,b.getsize()-1),random.randint(0,b.getsize()-1)
#                 statep = function.move(board,playerx,playery,"x")
# =============================================================================
            
            board=player.move(board,agent.other_symbol)
            
            #print(board)
            #board = player.move(board)
            s=function.score(board,agent.symbol,agent.other_symbol)
            if s==-1:
                agentloss+=1
                agent.fallback(board,s)
                player.fallback(board,s)
                #agent.fallback(board,s)
                #print(agent.states)
                print("Player win"+str(agentloss))
                #print(s)
                #print(board)

                board = b.newgame()
                break
            
            if s==0 and "_" not in board:
                agent.fallback(board,s)
                player.fallback(board,s)
                #agent.fallback(board,s)
                #print(agent.states)
                draw+=1
                print("Draw"+str(draw))
                board = b.newgame()
                break
            
            board = agent.move(board,agent.symbol)
            #board = mmAgent.move(board)
            #board = agent.move(board)
            #print(board)
            s=function.score(board,agent.symbol,agent.other_symbol)
            #print(board)
            
            

            if s==1:
                agentwin+=1
                agent.fallback(board,s)
                player.fallback(board,s)
                #agent.fallback(board,s)
                #print(agent.states)
                print("Agent win"+str(agentwin))
                board = b.newgame()
                break
            
            if s==0 and "_" not in board:
                draw+=1
                agent.fallback(board,s)
                player.fallback(board,s)
                #agent.fallback(board,s)
                #print(agent.states)
                print("Draw"+str(draw))
                board = b.newgame()
                break
    return agentwin,agentloss,draw

def eva(board,player1,player2):
    p1w = []
    p2w = []
    draw = []
    game_count = []
    count = 0
    for i in range(100):
        p1,p2,d = Play(board,player1,player2,25)
         
        player2.memory.clear()
        p1w.append(p1)
        p2w.append(p2)
        draw.append(d)
        count += 1
        game_count.append(count)
    
    return game_count,p1w,p2w,draw
    
if __name__ == "__main__":
    tf.reset_default_graph()    
    #BLANK = "_"
    b = board(3)
    randplayer1 = baselineOpponent("x","o")
    randplayer2 = baselineOpponent("o","x")
#def AgentQLearning(board):
    #agent = QL()  
#DQNAgent = DQNAgent("dqn", 0.95,0.01,True,"o")
    mmAgent = ExpectMinMaxAgent("o","x",3,0.95,0.01)
    c,x,y,z=eva(b,randplayer1,mmAgent)
    p = plt.plot(c, z, 'r-', c, x, 'g-', c, y, 'b-')
    
