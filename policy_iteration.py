#!/usr/bin/env python

#MIT License
#Copyright (c) 2017 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#Example of the policy iteration algorithm.

import numpy as np
import random
import tkinter
from PIL import Image, ImageTk

# global scope for visualization
root = tkinter.Tk()
canvas = tkinter.Canvas()
images = []
def create_rectangle(x1, y1, x2, y2, **kwargs):
    if 'alpha' in kwargs:
        alpha = int(kwargs.pop('alpha') * 255)
        fill = kwargs.pop('fill')
        fill = root.winfo_rgb(fill) + (alpha,)
        image = Image.new('RGBA', (x2-x1, y2-y1), fill)
        images.append(ImageTk.PhotoImage(image))
        canvas.create_image(x1, y1, image=images[-1], anchor='nw')
    canvas.create_rectangle(x1, y1, x2, y2, **kwargs)

def action_to_str(a):
    if a == -1:
        return "DONE"
    elif a == 0:
        return "^"
    elif a == 1:
        return "<"
    elif a == 2:
        return "v"
    elif a == 3:
        return ">"
    return "#"

def return_policy_evaluation(p, u, r, T, gamma):
    for s in range(12):
        if not np.isnan(p[s]):
            v = np.zeros((1,12))
            v[0,s] = 1.0
            action = int(p[s])
            u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return u

def return_policy_evaluation_linalg(p, r, T, gamma):
    u = np.zeros(12)
    for s in range(12):
        if not np.isnan(p[s]):
            action = int(p[s])
            u[s] = np.linalg.solve(np.identity(12) - gamma*T[:,:,action], r)[s]
    return u

def return_expected_action(u, T, v):
    """Return the expected action.

    It returns an action based on the
    expected utility of doing a in state s, 
    according to T and u. This action is
    the one that maximize the expected
    utility.
    @param u utility vector
    @param T transition matrix
    @param v starting vector
    @return expected action (int)
    """
    actions_array = np.zeros(4)
    for action in range(4):
         #Expected utility of doing a in state s, according to T and u.
         actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return np.argmax(actions_array)

def print_policy(p, shape):
    """Print the policy on the terminal

    Using the symbol:
    * Terminal state
    ^ Up
    > Right
    v Down
    < Left
    # Obstacle
    """
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(p[counter] == -1): policy_string += " *  "            
            elif(p[counter] == 0): policy_string += " ^  "
            elif(p[counter] == 1): policy_string += " <  "
            elif(p[counter] == 2): policy_string += " v  "           
            elif(p[counter] == 3): policy_string += " >  "
            elif(np.isnan(p[counter])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

def take_action(curr_state, action, T):
    """Return the next state and the given current state and the action chosen

    """
    coin  = random.random()
    # coin = 0.5
    # 12 possible next states
    next_states = T[curr_state, : , int(action)]
    prob_counter = 0.0
    # randomly take next action based on weights
    for state, prob in enumerate(next_states):
        if coin < prob_counter + prob:
            return state
        prob_counter += prob
    return -1

def execute_policy(p, T, start, max_t):
    """Place an agent in the environment and generate a stream of actions

    """
    curr_state = start
    output = []
    # no longer than max_t steps
    for i in range(max_t):
        output.append(int(p[curr_state]))
        if p[curr_state] == -1:
            break
        curr_state = take_action(curr_state, p[curr_state], T)
    return output

def main_iterative():
    """Finding the solution using the iterative approach

    """
    gamma = 0.999
    iteration = 0
    T = np.load("T.npy")

    #Generate the first policy randomly
    # Nan=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1

    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])

    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    while True:
        iteration += 1
        epsilon = 0.0001
        #1- Policy evaluation
        u1 = u.copy()
        u = return_policy_evaluation(p, u, r, T, gamma)
        #Stopping criteria
        delta = np.absolute(u - u1).max()
        if delta < epsilon * (1 - gamma) / gamma: break
        for s in range(12):
            if not np.isnan(p[s]) and not p[s]==-1:
                v = np.zeros((1,12))
                v[0,s] = 1.0
                #2- Policy improvement
                a = return_expected_action(u, T, v)         
                if a != p[s]: p[s] = a
        print_policy(p, shape=(3,4))

    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3,4))
    print("===================================================")
    print("=================== EXEC  POLICY ==================")
    start_pos = 11
    obs = execute_policy(p, T, start_pos, 12)
    s = "["
    for a in obs:
        s += action_to_str(a) + ", "
    if len(s) > 1:
        print(s[:-2] + "]")
    else:
        print("[]")
    print("====================== VITERBI ====================")
    # obs needs positive indices for viterbi alg implementation below
    obs = [obs[i]+1 for i in range(len(obs))]
    states = [i for i in range(12)]
    start_p = [0.0 for i in range(12)]
    start_p[start_pos] = 1.0

    # Viterbi needs 12x12 transition matrix
    # Generate the one induced by the policy
    trans_p = []
    for i in range(12):
        trans_p.append([0.0 for j in range(12)])
        if not np.isnan(p[i]) and not p[i] == -1:
            for j in range(12):
                trans_p[i][j] = T[i, j, int(p[i])]
    # emmission probabilities are induced by the policy
    emit_p = []
    for i in range(12):
        emit_p.append([0.0 for j in range(5)])
        # TODO - make nondeterministic policy possible
        if not np.isnan(p[i]):
            emit_p[i][int(p[i])+1] = 1.0
    viterbi(obs, states, start_p, trans_p, emit_p)


def main_linalg():
    """Finding the solution using a linear algebra approach

    """
    gamma = 0.999
    iteration = 0
    T = np.load("T.npy")

    #Generate the first policy randomly
    # Nan=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1

    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])

    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    while True:
        iteration += 1
        epsilon = 0.0001
        #1- Policy evaluation
        #u1 = u.copy()
        u = return_policy_evaluation_linalg(p, r, T, gamma)
        #Stopping criteria
        #delta = np.absolute(u - u1).max()
        #if (delta < epsilon * (1 - gamma) / gamma) or iteration > 100: break
        unchanged = True
        for s in range(12):
            if not np.isnan(p[s]) and not p[s]==-1:
                v = np.zeros((1,12))
                v[0,s] = 1.0
                #2- Policy improvement
                a = return_expected_action(u, T, v)         
                if a != p[s]: 
                    p[s] = a
                    unchanged = False
        print_policy(p, shape=(3,4))

        if unchanged: break

    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    #print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3,4))
    print("===================================================")

# Notes: emission probabilities are induced by the policy - 1.0 and 0.0 x3

# SOURCE: https://en.wikipedia.org/wiki/Viterbi_algorithm
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    # TODO - modification? For us, first observation is no different than the second
    for st in states:
        V[0] [st] = {"prob": start_p[st] * emit_p[st] [obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        prob_sum = 0.0
        for st in states:
            max_tr_prob = V[t - 1] [states[0]] ["prob"] * trans_p[states[0]] [st]
            # print("max prob = ", max_tr_prob)
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1] [prev_st] ["prob"] * trans_p[prev_st] [st]
                # print("testing against", tr_prob)
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t] [st] = {"prob": max_prob, "prev": prev_st_selected}
            prob_sum += max_prob
        
        # Update probabilities to sum to 1.0 at each time
        for st in states:
            V[t] [st] ["prob"] /= prob_sum

    # Back-propagate the answers
    # ASSUMES YOU KNOW YOU ENDED AT A GOAL STATE
    # Is this like running a version of viturbi backwards now?
    # 1. re-weight previous probabilities based on new info
    #       Easy way - when pr=1.0 at time=t, eliminate non-adjacent states at t-1 and so on
    #       What if not pr=1.0 at time=t? TODO - think about this.
    # 2. update to sum to 1.0
    for t in range(len(obs) - 1, 0, -1):
        for st in states:
            if V[t] [st] ["prob"] > 0.999999 and V[t] [st] ["prob"] < 1.000001:
                prob_sum = 0.0
                for prev_st in states:
                    # all non-adjacent states probability = 0
                    if trans_p[prev_st] [st] < 0.000001:
                        V[t-1] [prev_st] ["prob"] = 0.0
                    prob_sum += V[t-1] [prev_st] ["prob"]
                # Update probabilities to sum to 1.0
                for prev_st in states:
                    V[t-1] [prev_st] ["prob"] /= prob_sum

    for line in dptable(V):
        print(line)
    dptableTkinterAllTime(V)
    # dptableTkinterIterativeTime(V)

    opt = []
    max_prob = 0.0
    best_st = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    path_prob = max_prob
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1] [previous] ["prev"])
        path_prob *= V[t + 1] [previous] ["prob"]
        previous = V[t + 1] [previous] ["prev"]

    print ("The steps of states are ", opt, " with highest probability of ", path_prob)

def dptable(V):
    # Print a table of steps from dictionary
    yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%lf" % v[state] ["prob"]) for v in V)

# visualization over all time 
# (darkest where there are loops over time)
def dptableTkinterAllTime(V):
    scale = 60
    condensed_table = [0.0 for i in range(12)]
    for v in V:
        for (state_idx, _dict) in v.items():
            condensed_table[state_idx] += _dict["prob"]
    max_heat = 0.0
    for i in range(12):
        max_heat = max(max_heat, condensed_table[i])
    # max alpha is 1.0
    for i in range(12):
        condensed_table[i] /= max_heat

    y_ctr = 0
    for row in range(3):
        x_ctr = 0
        for col in range(4):
            a = condensed_table[4*row + col]
            create_rectangle(x_ctr, y_ctr, x_ctr+scale, y_ctr+scale, fill='red', alpha=a)
            x_ctr += scale
        y_ctr += scale

    canvas['bg'] = 'white'
    canvas.pack()

# visualization of states at each time step
def dptableTkinterIterativeTime(V):
    scale = 20
    x_ctr = 0
    for v in V:
        y_ctr = 0
        for _dict in v.values():
            create_rectangle(x_ctr, y_ctr, x_ctr+scale, y_ctr+scale, fill='red', alpha=_dict["prob"])
            y_ctr += scale
        x_ctr += scale
    canvas['bg'] = 'white'
    canvas.pack()

def main():

    main_iterative()
    #main_linalg()
    root.mainloop()


if __name__ == "__main__":
    main()