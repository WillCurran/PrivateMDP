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
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.special import rel_entr, kl_div

import Forward_Backward_Algiorithm_wikipedia as fb
import Viterbi_Algorithm_wikipedia as vt
import helpers as hlp

def return_policy_evaluation(p, u, r, T, gamma):
    for s in range(12):
        if not np.isnan(p[s]):
            v = np.zeros((1,12))
            v[0,s] = 1.0
            action = int(p[s])
            u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
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
    actions_array = np.zeros(2)
    for action in range(2):
        #Expected utility of doing a in state s, according to T and u.
        actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return np.argmax(actions_array)

def print_policy(p, shape):
    """Print the policy on the terminal

    Using the symbol:
    * Terminal state
    > Forward
    | Correct
    # Obstacle
    """
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if p[counter] == -1:
                policy_string += " *  "
            elif p[counter] == 0:
                policy_string += " >  "
            elif p[counter] == 1:
                policy_string += " |  "
            elif np.isnan(p[counter]):
                policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

def main_iterative(obs = []):
    """Finding the solution using the iterative approach

    """
    gamma = 0.999
    iteration = 0
    print("================= LOADING TRANSITIONAL MATRIX ==================")
    T = np.load("T2.npy")
    print(T)
    print("====================== POLICY ITERATION ========================")
    #Generate the first policy randomly
    # Nan=Obstacle, -1=Terminal, 0=Forward, 1=Correct
    p = np.random.randint(0, 2, size=(12)).astype(np.float32)
    #Obstacles
    #p[5] = np.NaN
    #Terminal States
    p[7] = -1

    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])

    #Reward vector
    r = np.array([-0.04, -0.04, -0.04, -0.04,
                  -0.04, -0.04, -0.04, +1.0,
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

    print("================ POLICY ITERATION FINAL RESULT =================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("================================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("================================================================")
    print_policy(p, shape=(3,4))
    print("===================== A Priori Analysis ======================")
    print("==================MDP + Policy = Markov Chain ==================")
    print("Policy: ")
    print([int(i) for i in p])
    print("Markov Chain:")
    markov_chain = hlp.to_markov_chain([int(i) for i in p], T, 12)
    markov_chain_df = pd.DataFrame(markov_chain)
    print(markov_chain_df)

    #set terminal state
    markov_chain[7][7]=1.0
    #set start state
    state = [
        [0.0,0.0,0.0,0.0,
         1.0,0.0,0.0,0.0,
         0.0,0.0,0.0,0.0]]
    state_history = [state[0]]
    for x in range(20):
        next_state = [[sum(a * b for a, b in zip(state_row, markov_chain_col))
                        for markov_chain_col in zip(*markov_chain)]
                                for state_row in state]
        state_history.append(next_state[0])
        state = next_state
    print("Stationary Distribution:")
    print(state)
    state_history_df = pd.DataFrame(state_history)
    #state_history_df.plot()
    #plt.show()
    print(state_history_df)
    print("================================================================")
    start_pos = 4
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
    print("========================= Dijkstra's =========================")
    # print("Distances")
    # print(distances)
    # print(trans_p)
    interesting_time = 4
    interesting_state = 3
    prior_expected_visits = hlp.get_expected_visits(states, start_p, T, p, interesting_time)
    print("Expected visits: \n" + ', '.join(["%.2f" % prior_expected_visits[st] for st in states]))
    print("Sum of expected visits should = 1 + t. %.2f == %d." % (sum(prior_expected_visits), 1+interesting_time) )
    if not obs:
        print("====================== Executing Policy ======================")
        obs = hlp.execute_policy(p, T, start_pos, 12)
        #obs = [0,0,1,-1]
    else:
        print("===================== Not Executing Policy ===================")
    print(obs)
    s = "["
    for a in obs:
        s += hlp.action_to_str2(a) + ", "
    if len(s) > 1:
        print(s[:-2] + "]")
    else:
        print("[]")
    # obs needs positive indices for viterbi alg implementation below
    #obs_original = obs
    obs = [obs[i]+1 for i in range(len(obs))]
    
    end_state = 7
    trans_p[end_state][end_state]= 1.0 #setting terminal state?
    
    trans_p_df = pd.DataFrame(trans_p)
    emit_p_df = pd.DataFrame(emit_p)
    print("##OBSERVATIONS##")
    print(obs)
    print("##STATES##")
    print(states)
    print("##STARTING DISTRIBUTION##")
    print(start_p)
    print("##TRANSITION DISTRIBUTION##")
    print(trans_p_df)
    print("##EMISSIONS DISTRIBUTION##")
    print(emit_p_df)
    print("=========================== VITERBI ==========================")
    (dp_table, max_path_prob) = vt.viterbi_custom(obs, states, start_p, trans_p, emit_p)
    if interesting_time >= len(dp_table):
        print("Actual execution did not go as long as %d steps. How to handle information gain here?" % interesting_time)
    else:
        post_expected_visits = [dp_table[interesting_time][st]["prob"] for st in states]
        print("Actual expected visits given single execution: \n" + ', '.join(["%.2f" % post_expected_visits[st] for st in states]))
        print("====================== INFORMATION GAIN ====================")
        #ig = hlp.information_gain(prior_expected_visits, post_expected_visits, interesting_state, max_path_prob)
        #print("Information Gain on state=%d and time=%d: %.2f" % (interesting_state, interesting_time, ig))
    print("=========================== Forward Backward ==========================")
    result = fb.fwd_bkw_custom(obs, states, start_p, trans_p, emit_p, end_state)
    for line in result:
        print(*line)
    print('##FORWARD##')
    print(pd.DataFrame(result[0]))
    print('##BACKWARD##')
    print(pd.DataFrame(result[1]))
    print('##POSTERIOR##')
    print(pd.DataFrame(result[2]))
    
    print("==================== KL Divergence/Relative Entropy ===================")
    #p = [i.values() for i in result[2]]
    #convert dictionary to list
    p = []
    for i in range(len(obs)):
    	val_ls = []
    	for key, val in result[2][i].items():
    		val_ls.append(val)
    	p.append(val_ls)
    q = state_history[:len(obs)]
    
    #https://stackoverflow.com/questions/63369974/3-functions-for-computing-relative-entropy-in-scipy-whats-the-difference
    print('##ACTUAL DISTRIBUTION##')
    print(pd.DataFrame(p))
    print('##REFERENCE DISTRIBUTION##')
    print(pd.DataFrame(q))
    print('##ENTROPY##')
    for i in range(len(p)):
        print(entropy(p[i], q[i]))
    print('##RELATIVE ENTROPY##')
    for i in range(len(p)):
        print(rel_entr(p[i], q[i]))
        print(sum(rel_entr(p[i], q[i])))
    print('##KL DIVERGENCE##')
    for i in range(len(p)):
        print(kl_div(p[i], q[i]))
        print(sum(kl_div(p[i], q[i])))

def main():
    main_iterative()
    #main_linalg()

if __name__ == "__main__":
    main()
