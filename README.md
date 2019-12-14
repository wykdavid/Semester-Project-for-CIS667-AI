# Semester-Project-for-CIS667-AI
This project should run on Python 3
Requires tensorflow 1.13.1
Requires GPU

Execution: Run the py script named game, it will execute 100 iterations of battles between the AI agent and a baseline opponent in a 3X3 board. Each battle contains 100 games. The final result is a plot showing the win rate of the agent and baseline opponent, and the draw rate. The green, blue, red represent agent, baseline opponent, and draw respectively.

Notice: The result is produced in board size 3. It will still work in board size 4, 5, 6 and 7. However, when working on board size bigger than 3, it will cost a lot of time to execute game and train the neural network model (in size 4 needs more than 1 hour, size 5 more than 5 hours, size 6 more than 9 hours and size 7 more than 13 hours). If you want to examine the result in different sizes, you need to go to game file, find line 101, change the value of problem_size to the value you want. If you want to see the plot for the number of nodes searched, you should go to line 102 and set the value of game_number 1. In this project, agent always goes second

