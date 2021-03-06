from Tkinter import Tk, Frame, BOTH, Button, Label
import tkFont
from state import *


class gui(Frame):

    def __init__(self, window, state_a, ia, **kwargs):
        Frame.__init__(self, window, width=768, height=576, **kwargs)
        self.state_a = state_a
        self.pack(fill=BOTH)
        self.buttons = []
        self.ia = ia
        self.scores = [0, 0]
        self.scores_lbl = []
        self.scores_lbl.append(Label(self, text=str(self.scores[0])))
        self.scores_lbl.append(Label(self, text=str(self.scores[1])))
        self.font = tkFont.Font(family='Helvetica', size=36, weight='bold')
        action_possible = self.state_a.get_possible_action()
        txt_button = [str(self.state_a.board[i]) for i in reversed(
            range(6))] + [str(self.state_a.board[i]) for i in range(6, 12)]

        for i in range(12):
            self.buttons.append(
                Button(self, text=txt_button[i],
                       command=lambda x=i: self.click(x)))
            self.buttons[i].grid(row=i // 6, column=i % 6)
            self.buttons[i].config(height=2, width=2, font=self.font)
            if i in action_possible:
                self.buttons[i].config(state='normal')
            else:
                self.buttons[i].config(state='disabled')
        self.scores_lbl[0].grid(row=0, column=6)
        self.scores_lbl[1].grid(row=1, column=6)
        self.scores_lbl[0].config(height=2, width=2, font=self.font)
        self.scores_lbl[1].config(height=2, width=2, font=self.font)

        self.scores[
            0] += self.state_a.get_next_state(self.ia.return_action(self.state_a))[2]
        self.state_a = self.state_a.get_next_state(
            self.ia.return_action(self.state_a))[1]

        self.update_gui()

    def click(self, i):
        if i in range(6):
            i = 5 - i

        self.scores[1] += self.state_a.get_next_state(i)[2]
        self.state_a = self.state_a.get_next_state(i)[1]
        self.update_gui()

        self.scores[
            0] += self.state_a.get_next_state(self.ia.return_action(self.state_a))[2]
        self.state_a = self.state_a.get_next_state(
            self.ia.return_action(self.state_a))[1]
        self.update_gui()

    def update_gui(self):
        if not self.state_a.is_terminated():
            action_possible = self.state_a.get_possible_action()
            txt_button = [str(self.state_a.board[j]) for j in reversed(
                range(6))] + [str(self.state_a.board[j]) for j in range(6, 12)]
            for j in range(12):
                self.buttons[j].config(text=txt_button[j])
                if j in action_possible:
                    if j in range(6):
                        self.buttons[5 - j].config(state='normal')
                    else:
                        self.buttons[j].config(state='normal')
                else:
                    if j in range(6):
                        self.buttons[5 - j].config(state='disabled')
                    else:
                        self.buttons[j].config(state='disabled')
            self.scores_lbl[0].config(text=str(self.scores[0]))
            self.scores_lbl[1].config(text=str(self.scores[1]))
        else:
            print "Winner : " + str(winner(self.state_a, self.scores[0], self.scores[1]))
            for i in range(12):
                self.buttons[i].destroy()
            self.scores_lbl[0].destroy()
            self.scores_lbl[1].destroy()
            self.grid_forget()
            self.end_lbl = Label(self, text="Winner : " +
                                 str(winner(self.state_a, self.scores[0], self.scores[1])))
            self.end_lbl.grid(row=0, column=0)
            self.end_lbl.config(height=1, width=10, font=self.font)
