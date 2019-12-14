# Semester-Project-for-CIS667-AI
This project should run on Python 3
Requires tensorflow 1.13.1
Requires GPU

Execution: Download all py files and put them in one directory. Run the py script named game, it will execute 100 iterations of battles between the AI agent and a baseline opponent in a 3X3 board. Each battle contains 100 games. The final result is a plot showing the win rate of the agent and baseline opponent, and the draw rate. The green, blue, red represent agent, baseline opponent, and draw respectively.

Notice: In this project, agent always goes second. The result is generated from a 3x3 board. It will still work in board size 4, 5, 6 and 7. However, when working on board size bigger than 3, it will cost a lot of time to execute game and train the neural network model (in size 4 needs more than 1 hour, size 5 more than 5 hours, size 6 more than 9 hours and size 7 more than 13 hours). 

If you want to examine the result in different sizes, you need to go to game file, find line 101, change the value of problem_size to the value you want. 

If you want to see the plot for the number of nodes searched, you should go to line 102 and set the value of game_number 1. 

If you want to see the result of baseline vs agent only use expectminimax algorithm, you should go to line 104 and change the value of variable depth_limit to a value that greater than the square of current board size(for example, change to 9 when board size is 3). The variable depth_limit can also be used to change the depth limited in the tree search algorithm.

