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
import pandas as pd
import random
import math
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

def dijkstra(trans_p, start, goal):
    distances = [math.inf for i in range(12)]
    previous = [math.nan for i in range(12)]
    start = 4
    min_heap = [(0,start)]
    distances[start] = 0
    previous[start] = -1
    goal = 2
    max_time = 11
    while len(min_heap) != 0:
        value, curr_state = heapq.heappop(min_heap)
        if curr_state == goal:
            break
        for i in range(12):
            prob = trans_p[curr_state][i]
            if prob != 0:
                distance = - math.log(prob)
                alt = distances [curr_state] + distance
                if alt < distances[i]:
                    distances[i] = alt
                    previous[i] = curr_state
                    heapq.heappush(min_heap,(alt, i))

    return previous

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
    print("Stationary Distribution")
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
    print(obs)
    print(states)
    print(start_p)
    print(trans_p_df)
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

def information_gain(Q, P, s, path_probability):
    """Calculate a bound for expected(?) information gain given one execution of viturbi on one string of observations
    Q is the "prior" distribution over all states for a certain time
    P is the "posterior" distribution over all states for a certain time
    s is the state we are interested in
    path_probability is the probability of a path (pr of observation string * pr of most likely path given by viterbi)"""

    # calculate information gain (relative entropy, or Kullback–Leibler divergence)
    # D_{kl}(P || Q) = sum over all outcomes of Pr(outcome)*log2(P(outcome)/Q(outcome))

    rel_entropy = 0.0
    # If Q[s] == 0 or Q[s] == 1, then there's no uncertainty
    if Q[s] < CMP_DELTA or Q[s] > 1.0 - CMP_DELTA:
        return (0.0, 0.0)
    # If P[s] = 0, then math works out ok, since 0log0 == 0
    elif P[s] < CMP_DELTA:
        rel_entropy = (1.0 - P[s]) * math.log2((1.0 - P[s]) / (1.0 - Q[s]))
    # otherwise, use standard formula
    else:
        rel_entropy = P[s] * math.log2(P[s] / Q[s]) + \
            (1.0 - P[s]) * math.log2((1.0 - P[s]) / (1.0 - Q[s]))

    max_remaining_info = 0.0
    if Q[s] > 0.5:
        # max info is when P=0.0
        max_remaining_info = (1.0 - 0.0) * math.log2(1.0 / Q[s])
    else:
        # max info is when P=1.0
        max_remaining_info = 1.0 * math.log2(1.0 / Q[s])

    # what we know based on paths we tested.
    known_rel_entropy = rel_entropy
    # worst case expected entropy, given what we know.
    worst_expected_entropy = path_probability * rel_entropy + (1.0 - path_probability) * max_remaining_info
    return (known_rel_entropy, worst_expected_entropy)

def get_expected_visits(states, start_p, T, p, t):
    """Get number of extpected visits of each state after t steps
    with no information about observations

    states : state indices
    start_p : initial probability distribution
    T : original transition matrix
    p : policy
    t : time interval
    """
    # build new transition matrix from policy
    trans_p = []
    for i in range(12):
        trans_p.append([0.0 for j in range(12)])
        if not np.isnan(p[i]) and not p[i] == -1:
            for j in range(12):
                trans_p[i][j] = T[i, j, int(p[i])]
        elif p[i] == -1:
            # if at a terminal, then consider that you are at this state for all remaining time
            trans_p[i][i] = 1.0

    # initial distribution tells us where we will be at time=0
    curr_p = [start_p[j] for j in range(12)]
    print("time=%d : %s" % (0, ', '.join(["%.2f" % curr_p[st] for st in states]) + ": sum=%.2f" % sum(curr_p)))
    for i in range(1,t+1):
        next_p = [0.0 for j in range(12)]
        for st in states:
            for next_st in states:
                next_p[next_st] += curr_p[st] * trans_p[st][next_st]
        for st in states:
            curr_p[st] = next_p[st]
        print("time=%d : %s" % (i, ', '.join(["%.2f" % curr_p[st] for st in states]) + ": sum=%.2f" % sum(curr_p)))
    return curr_p

def generate_naive_paths():
    """Generate all action sequences, then narrow down
    based on 

    Goal: gain intuition on how to create a smarter generator of observation strings
    """
    pass

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
    return (V, path_prob)

def dptable(V):
    # Print a table of steps from dictionary
    yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%lf" % v[state] ["prob"]) for v in V)

def to_markov_chain(p, T, max_t):
    result = []
    for t in range(max_t):
        result.append([row[p[t]] for row in T[t][:]])
    return result

# Source: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    """Forward–backward algorithm."""
    # Forward part of the algorithm
    fwd = []
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k] * trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)

    # Backward part of the algorithm
    bkw = []
    for i, observation_i_plus in enumerate(reversed(observations[1:] + [None,])):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)

    # Merging the two parts
    posterior = []
    for i in range(len(observations)):
    	#this was added because the state transition distribution had rows with all 0s for the terminal state.
    	#you have to set 1.0 to the terminal state when transitioning from the terminal state
    	#if not p_fwd:
    	#	posterior.append({st: 0.0 for st in states})
    	#else:
    	#	posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})
    	posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})
    print(p_fwd)
    print(p_bkw)
    #https://davidamos.dev/the-right-way-to-compare-floats-in-python/
    assert math.isclose(p_fwd, p_bkw)     
    return fwd, bkw, posterior

def main():
    main_iterative()
    #main_linalg()

if __name__ == "__main__":
    main()
