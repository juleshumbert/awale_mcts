from state import *
import random
import math
from csv import *


class Node():
    """Documentation for node

    """

    def __init__(self, ident, father, children, state, reward, n_visit):
        self.ident = ident
        self.father = father
        self.children = children
        self.state = state
        self.n_visit = n_visit
        self.reward = reward

    def __str__(self):
        return "\nIdent : " + str(self.ident) + "\nFather : " + str(self.father) + "\nSons : " \
            + str(self.children) + "\nState : " + str(self.state) + \
            "\nN_visit : " + str(self.n_visit) + \
            "\nReward : " + str(self.reward)


class random_AI():
    """Random AI
    """

    def __init__(self):
        self.activ = True

    def return_action(self, current_state):
        actions = current_state.get_possible_action()
        action_chosen = actions[random.randint(0, len(actions) - 1)]
        rew = [current_state.get_next_state(a)[2] for a in actions]
        rew = [float(i) + 0.1 for i in rew]
        m = sum(rew)
        pick = random.uniform(0, m)
        current = 0
        for i in range(len(rew)):
            value = rew[i]
            current += value
            if current > pick:
                return actions[i]


class minimax_AI():
    """Random AI
    """

    def __init__(self):
        self.activ = True

    def return_action(self, current_state):
        actions = current_state.get_possible_action()
        next_states = [current_state.get_next_state(a)[1] for a in actions]
        next_reward = [10 * current_state.get_next_state(
            a)[2] + current_state.get_next_state(a)[3] for a in actions]
        rew = []
        for s in next_states:
            next_actions = s.get_possible_action()
            next_rew = [s.get_next_state(a)[2] for a in next_actions]
            if len(next_actions) != 0:
                rew.append(next_reward[next_states.index(s)] - max(next_rew))
            else:
                rew.append(-1000)
        rew = [(float(i) + 0.1 + min(rew) * math.copysign(1, min(rew)))
               for i in rew]
        m = sum(rew)
        pick = random.uniform(0, m)
        current = 0
        for i in range(len(rew)):
            value = rew[i]
            current += value
            if current > pick:
                return actions[i]


class mcts():
    """MCTS AI

    """

    def __init__(self, player=1):
        self.c_value = 1.4
        self.path = []
        self.player = player
        self.tot_sim = 0
        self.monte_carlo = False
        self.file_name_nodes = 'mcts_nodes.csv'
        self.file_name_n_sim = 'mcts_n_sim.csv'
        self.minimax = minimax_AI()
        self.sizes = [[] for _ in range(49)]

    def load_ai(self):
        self.tree = []
        f1 = open(self.file_name_nodes, 'rb')
        f2 = open(self.file_name_n_sim, 'rb')
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        for r in reader2:
            self.tot_sim = int(r[0])
        for r in reader1:
            row = [int(c) for c in r]
            state = state_awale(row[1:13], row[13])
            n = Node(row[0], row[22:], row[16:22],
                     state, row[14], row[15])
            self.tree.append(n)
        self.path.append(self.tree[1])
        f1.close()
        f2.close()

        for n in self.tree:
            self.sizes[sum(n.state.board)].append(n.ident)

    def write_ai(self):
        f1 = open(self.file_name_nodes, 'w')
        f2 = open(self.file_name_n_sim, 'w')

        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer2.writerow((str(self.tot_sim),))
        for n in self.tree:
            line = n.ident,
            line += tuple(n.state.board)
            line += (n.state.player, n.reward, n.n_visit)
            line += tuple(n.children)
            line += tuple(n.father)
            writer1.writerow(line)
        f1.close()
        f2.close()

    def return_action(self, current_state):
        state_known = self.state_already_known(current_state)

        if state_known:
            node = state_known[1]
            if node.children == [-1] * 6:
                self.monte_carlo = True
                return self.expansion(node)
            else:
                return self.selection(node)
        elif not self.monte_carlo:
            n = Node(self.tree[-1].ident + 1, (self.path[-1].ident,),
                     [-1] * 6, current_state, 0, 0)
            self.sizes[sum(n.state.board)].append(n.ident)
            self.tree.append(n)
            self.monte_carlo = True
            return self.expansion(n)
        else:
            return self.monte_carlo_end(current_state)

    def selection(self, n):
        children = [self.tree[i]
                    for i in filter(lambda x: x != -1, n.children)]
        uct = []
        for c in children:
            uct.append(float(c.reward) / (c.n_visit + 1) + self.c_value *
                       math.sqrt(math.log(self.tot_sim) / (c.n_visit + 1)))

        index = uct.index(max(uct))
        chosen_node = children[index]
        self.path.append(chosen_node)
        return self.get_action_for_state(n.state, chosen_node.state)

    def expansion(self, n):
        next_nodes = []
        possible_actions = n.state.get_possible_action()
        for i in range(len(possible_actions)):
            a = possible_actions[i]
            next_state = n.state.get_next_state(a)[1]
            state_known = self.state_already_known(next_state)
            if state_known:
                next_nodes.append(state_known[1])
                state_known[1].father = list(state_known[1].father)
                state_known[1].father.append(n.ident)
                n.children[i] = state_known[1].ident
            else:
                n.children[i] = self.tree[-1].ident + 1
                new_node = Node(self.tree[-1].ident + 1, (n.ident,),
                                [-1] * 6, next_state, 0, 0)
                next_nodes.append(new_node)
                self.sizes[sum(new_node.state.board)].append(new_node.ident)
                self.tree.append(new_node)
        r = random.randint(0, len(next_nodes) - 1)
        self.path.append(next_nodes[r])
        return self.get_action_for_state(n.state, next_nodes[r].state)

    def monte_carlo_end(self, current_state):
        actions = current_state.get_possible_action()
        rewards = [current_state.get_next_state(a)[2] for a in actions]
        action_chosen = actions[self.weighted_random_action(rewards)]
        action_chosen = self.minimax.return_action(current_state)
        return action_chosen

    def weighted_random_action(self, rew):
        rew = [float(i) + 0.1 for i in rew]
        m = sum(rew)
        pick = random.uniform(0, m)
        current = 0
        for i in range(len(rew)):
            value = rew[i]
            current += value
            if current > pick:
                return i

    def back_propagation(self, win):
        self.tot_sim += 1
        for node in self.path:
            node.reward += win
            node.n_visit += 1
        self.path = []
        self.monte_carlo = False
        self.path.append(self.tree[1])

    def get_action_for_state(self, prev_state, next_state):
        possible_actions = prev_state.get_possible_action()
        for a in possible_actions:
            if prev_state.get_next_state(a)[1] == next_state:
                return a

    def state_already_known(self, state):
        res = False
        for i in self.sizes[sum(state.board)]:
            if self.tree[i].state == state:
                return True, self.tree[i]
        if not res:
            return res
# mc = mcts()
# mc.load_ai()
# current_state = state_awale()
# current_state.board = [0, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4]
# current_state.player = 1
# state_known = mc.state_already_known(current_state)
# print current_state
# for n in mc.tree:
#     print n
# print '\n\n\n'
# if state_known:
#     node = state_known[1]
#     mc.path.append(node)
#     if node.children == [-1] * 6:
#         mc.expansion(node)

#     else:
#         mc.selection(node)
# for n in mc.tree:
#     print n
