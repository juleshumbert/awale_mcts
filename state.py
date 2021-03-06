
import csv


class state_awale():
    """This class is representing a state in the awale game

    """

    def __init__(self, board=[2] * 12, player=0):
        self.board = board
        self.player = player

    def __str__(self):
        return '\n' + str([self.board[i] for i in reversed(range(6))]) \
            + '\n' + str([self.board[i] for i in range(6, 12)]) \
            + '\nPlayer : ' + str(self.player)

    def __eq__(self, other):
        if self.board == other.board and self.player == other.player:
            return True
        # elif self.player != other.player \
        #         and self.board[0:6] == other.board[6:13]\
        #         and other.board[0:6] == self.board[6:13]:
        #     return True
        else:
            return False

    def is_terminated(self):
        if sum(self.board[i] for i in range(6)) == 0:
            return True, 1
        elif sum(self.board[i] for i in range(6, 12)) == 0:
            return True, 0
        else:
            return False

    def get_next_state(self, action):
        n_seeds_init = self.board[action]
        n_seeds = n_seeds_init
        res = list(self.board)
        res[action] = 0
        i = 1
        direct_reward2 = 0

        while n_seeds > 0:
            if i % 12 != 0:
                res[(action + i) % 12] += 1
                n_seeds -= 1
                if i % 12 in range(((self.player + 1) % 2) * 6,
                                   ((self.player + 1) % 2) * 6 + 6):
                    direct_reward2 -= 1
            i += 1

        direct_reward = 0
        j = (action + n_seeds_init) % 12
        while (j in range(((self.player + 1) % 2) * 6,
                          ((self.player + 1) % 2) * 6 + 6))\
                and (res[j] == 2 or res[j] == 3):
            direct_reward += res[j]
            res[j] = 0
            j -= 1

        if self.board[action] == 0 or \
           action not in range(self.player * 6, self.player * 6 + 6):
            return False

        elif sum(res[i] for i in range(((self.player + 1) % 2) * 6,
                                       ((self.player + 1) % 2) * 6 + 6)) == 0 \
                and sum(res[i] for i in range(12)) != 0 \
                and len([i for i, e in enumerate(range(((self.player) % 2) * 6,
                                                       ((self.player) % 2) * 6 + 6)) if self.board[e] != 0]) != 1\
                and self.board != [1, 0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 1] \
                and self.board != [1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1]:

            return False

        else:
            return True, state_awale(res, (self.player + 1) % 2,), direct_reward, direct_reward2

    def get_possible_action(self):
        possible_actions = []
        for i in range(self.player * 6, self.player * 6 + 6):
            if self.get_next_state(i):
                possible_actions.append(i)
        return possible_actions


def winner(state, score_0, score_1):
    if state.is_terminated()[1] == 1:
        score_1 += sum(state.board[i] for i in range(6, 12))
    else:
        score_0 += sum(state.board[i] for i in range(6, 12))
    return [score_0, score_1].index(max([score_0, score_1]))

if __name__ == '__main__':
    n = state_awale([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 1)
    print n
    print n.get_possible_action()
    print n.get_next_state(5)
