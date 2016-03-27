This is an implementation of MCTS for Awalé game.

Awalé game is an abstract strategy game also called Oware. The rules can be seen on :
https://en.wikipedia.org/wiki/Oware

The AI implemented is an MCTS method with a randomized minimax method for the termination of the game instead of fully Monte Carlo.

To train the AI just run:

>>python game.py --train

The AI will play against the same randomized minimax method used for termination.

To play againt the AI run:

>>python game.py --gui

You will play againt the AI which will be player North (player 0) and you will play the south (player 1)

The AI has only be trained for a smaller game which initial state is:
[ 2 | 2 | 2 | 2 | 2 | 2 ]
[ 2 | 2 | 2 | 2 | 2 | 2 ]
Because for bigger configuration training time can be too long.

The total number of configuration is still :
664649732