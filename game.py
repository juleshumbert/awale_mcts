from Tkinter import Tk, Frame, BOTH, Button
import sys
from gui import *
from state import *
from ai import *
import numpy as np


if sys.argv[1] == '--gui':
    current_state = state_awale()
    mc = mcts()
    mc.load_ai()
    window = Tk()
    window.title(string='Awale')
    gui = gui(window, current_state, mc)
    gui.mainloop()

if sys.argv[1] == '--train':

    mc = mcts()
    mc.load_ai()
    tot = []
    for j in range(50):
        for i in range(1000):
            nb_action = 0
            score_0, score_1 = 0, 0
            state = state_awale()
            rd = minimax_AI()
            while not state.is_terminated():
                ac = mc.return_action(state)
                # print ac, state
                next_state = state.get_next_state(ac)
                score_0 += next_state[2]
                state = next_state[1]
                nb_action += 1
                if not state.is_terminated():
                    ac = rd.return_action(state)
                    # print ac, state
                    next_state = state.get_next_state(ac)
                    score_1 += next_state[2]
                    state = next_state[1]
                    if state.is_terminated():
                        mc.back_propagation(
                            int(winner(state, score_0, score_1) == 0))
                else:
                    mc.back_propagation(
                        int(winner(state, score_0, score_1) == 0))
            tot.append(winner(state, score_0, score_1))
            print j, i, winner(state, score_0, score_1)

            # print state
            # print score_0, score_1
            # print nb_action
        for i in range(len(mc.sizes)):
            print np.count_nonzero(np.array(mc.sizes)[i])
        mc.write_ai()
        print sum(tot)
