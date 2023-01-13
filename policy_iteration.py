#!/usr/bin/env python
# https://github.com/mpatacchiola/dissecting-reinforcement-learning
# MIT License
# Copyright (c) 2017 Massimiliano Patacchiola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Example of the policy iteration algorithm.

import math
import random
import statistics

from scipy.special import rel_entr, kl_div
from scipy.stats import entropy
from tabulate import tabulate

import Forward_Backward_Algiorithm_wikipedia as fb
import Viterbi_Algorithm_wikipedia as vt
import dijkstras as dk
import helpers as hlp
import hmm as hmm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def return_policy_evaluation(p, u, r, T, gamma):
    for s in range(12):
        if not np.isnan(p[s]):
            v = np.zeros((1, 12))
            v[0, s] = 1.0
            action = int(p[s])
            u[s] = r[s] + gamma * \
                np.sum(np.multiply(u, np.dot(v, T[:,:, action])))
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
        # Expected utility of doing a in state s, according to T and u.
        actions_array[action] = np.sum(
            np.multiply(u, np.dot(v, T[:,:, action])))
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
            if(p[counter] == -1):
                policy_string += " *  "
            elif(p[counter] == 0):
                policy_string += " ^  "
            elif(p[counter] == 1):
                policy_string += " <  "
            elif(p[counter] == 2):
                policy_string += " v  "
            elif(p[counter] == 3):
                policy_string += " >  "
            elif(np.isnan(p[counter])):
                policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)


def main_iterative(obs=[]):
    """Finding the solution using the iterative approach

    """
    gamma = 0.999
    iteration = 0
    print("================= LOADING TRANSITIONAL MATRIX ==================")
    T = np.load("T.npy")
    print(T)
    print("====================== POLICY ITERATION ========================")
    # Generate the first policy randomly
    # Nan=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    # Obstacles
    p[5] = np.NaN
    # Terminal States
    p[3] = p[7] = -1

    # Utility vectors
    u = np.array([0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0])

    # Reward vector
    r = np.array([-0.04, -0.04, -0.04, +1.0,
                  -0.04, 0.0, -0.04, -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    while True:
        iteration += 1
        epsilon = 0.0001
        # 1- Policy evaluation
        u1 = u.copy()
        u = return_policy_evaluation(p, u, r, T, gamma)
        # Stopping criteria
        delta = np.absolute(u - u1).max()
        if delta < epsilon * (1 - gamma) / gamma:
            break
        for s in range(12):
            if not np.isnan(p[s]) and not p[s] == -1:
                v = np.zeros((1, 12))
                v[0, s] = 1.0
                # 2- Policy improvement
                a = return_expected_action(u, T, v)
                if a != p[s]:
                    p[s] = a
        print_policy(p, shape=(3, 4))

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
    print_policy(p, shape=(3, 4))
    print("======================= A Priori Analysis =======================")
    print("================== MDP + Policy = Markov Chain ==================")
    print("Policy: ")
    policy = [np.NaN if np.isnan(i) else int(i) for i in p]
    print(policy)
    print("Markov Chain:")
    markov_chain = hlp.to_markov_chain(policy, T, 12)
    markov_chain_df = pd.DataFrame(markov_chain)
    print(markov_chain_df.to_string())
    # set obstacles to loop
    markov_chain[5][5] = 1.0
    # set terminal state to loop
    markov_chain[3][3] = 1.0
    markov_chain[7][7] = 1.0
    # set start state
    starting_state = [0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0, 0.0]
    state, state_history = hlp.state_probabilities_up_to_n_steps(markov_chain, starting_state, 20)
    print("State probabilities after 20 steps")
    print(state)
    state_history_df = pd.DataFrame(state_history)
    # state_history_df.plot()
    # plt.show()
    print(state_history_df.to_string())
    print("======================= Equilibrium Distribtuion of MDP =======================")
    
    if hlp.is_irreducible_aperiodic(T.sum(axis=-1)):
        print("The matrix is irreducible and aperiodic")
    else:
        print("The matrix is NOT irreducible and aperiodic")

    # Calculate equilibrium distribution
    print(np.shape(T))
    # equilibrium = hlp.equilibrium_distribution_2(T)
    equilibrium_dist = hlp.equilibrium_distribution_power_iteration_3d_cols_left(T, np.array(starting_state))
    print(equilibrium_dist)
    print("======================= Stationary Distribtuion of Markov Chain=======================")
    stationary_dist = hlp.stationary_distribution(np.array(markov_chain))
    print(stationary_dist)
    print("=========================== Create HMM ==========================")
    start_state = 8
    states, start_p, trans_p, emit_p = hlp.to_hidden_markov_model(T, p, 12, 4, start_state)
    print("============================ Dijkstra ===========================")
    # print(trans_p)

    # g = trans_to_graph(trans_p)
    # D = dijkstra(g,"v4","v7")
    end_state = 3
    D = (dk.dijkstra(trans_p, start_state, end_state, None, 3))
    print(D)
    print(dk.path_prob(D, trans_p))

    print("=========================== KDijkstra ===========================")

    A = dk.kdijkstra_actions(trans_p, start_state, end_state, 10, p, 10)

    print(*A, sep="\n")

    print("======================= Expected Visits ==========================")
    interesting_time = 4
    interesting_state = 3
    prior_expected_visits = hlp.get_expected_visits(
        states, start_p, T, p, interesting_time)
    print("Expected visits: \n" + 
          ', '.join(["%.2f" % prior_expected_visits[st] for st in states]))
    print("Sum of expected visits should = 1 + t. %.2f == %d." % 
          (sum(prior_expected_visits), 1 + interesting_time))
    if not obs:
        print("====================== Executing Policy ======================")
        obs = hlp.execute_policy(p, T, start_state, 12)
        print("Observations from policy execution")
        print(obs)
    else:
        print("===================== Not Executing Policy ===================")
    print("====================== Policy Translation ======================")
    obs = A[0][0]
    print(obs)

    s = "["
    for a in obs:
        s += hlp.action_to_str_russel_norvig_world(a) + ", "
    if len(s) > 1:
        print(s[:-2] + "]")
    else:
        print("[]")
    # obs needs positive indices for viterbi alg implementation below
    # obs_original = obs
    obs = [obs[i] + 1 for i in range(len(obs))]
    print(obs)
    print("=========================== Analyze HMM ==========================")
    # Set obstacle states to loop
    trans_p[5][5] = 1.0
    # Set Terminal states to loop
    trans_p[7][7] = 1.0
    end_state = 3
    trans_p[end_state][end_state] = 1.0
    trans_p_df = pd.DataFrame(trans_p)
    emit_p_df = pd.DataFrame(emit_p)
    print("##OBSERVATIONS##")
    print(obs)
    print("##STATES##")
    hlp.print_world(states, shape=(3, 4))
    print("##STARTING DISTRIBUTION##")
    print(start_p)
    print("##TRANSITION DISTRIBUTION##")
    print(trans_p_df.to_string())
    print("##EMISSIONS DISTRIBUTION##")
    print(emit_p_df.to_string())
    print("=========================== VITERBI ==========================")
    (dp_table, max_path_prob) = vt.viterbi_custom(
        obs, states, start_p, trans_p, emit_p)
    if interesting_time >= len(dp_table):
        print("Actual execution did not go as long as %d steps. How to handle information gain here?" % interesting_time)
    else:
        post_expected_visits = [
            dp_table[interesting_time][st]["prob"] for st in states]
        print("Actual expected visits given single execution: \n" + 
              ', '.join(["%.2f" % post_expected_visits[st] for st in states]))
        print("====================== INFORMATION GAIN ====================")
        # ig = hlp.information_gain(prior_expected_visits, post_expected_visits, interesting_state, max_path_prob)
        # print("Information Gain on state=%d and time=%d: %.2f" % (interesting_state, interesting_time, ig))
    print("=========================== FORWARD BACKWARD ==========================")
    # result = fb.fwd_bkw_custom(obs, states, start_p, trans_p, emit_p, end_state)
    # for line in result:
    #    print(*line)
    # print('##FORWARD##')
    # print(pd.DataFrame(result[0]).to_string())
    # print('##BACKWARD##')
    # print(pd.DataFrame(result[1]).to_string())
    # print('##POSTERIOR##')
    # print(pd.DataFrame(result[2]).to_string())

    # p = [i.values() for i in result[2]]
    # convert dictionary to list
    # p = []
    # for i in range(len(obs)):
    #    val_ls = []
    #    for key, val in result[2][i].items():
    #        val_ls.append(val)
    #    p.append(val_ls)
    print("observation sequence:")
    print(obs)
    russelhmm = hmm.HMM(np.array(trans_p), np.array(emit_p), np.array(start_p))
    posterior_marginals = russelhmm.forward_backward(obs)
    print("posterior marginals:")
    print(pd.DataFrame(posterior_marginals).to_string())
    print("========== Difference Between Prior and Posterior Marginals  ==========")
    p = posterior_marginals[1:]
    q = state_history[:len(obs)]

    rows = len(p)
    cols = len(p[0])
    # https://stackoverflow.com/questions/63369974/3-functions-for-computing-relative-entropy-in-scipy-whats-the-difference
    print('##ACTUAL DISTRIBUTION##')
    print(pd.DataFrame(p).to_string())
    print('##REFERENCE DISTRIBUTION##')
    print(pd.DataFrame(q).to_string())
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

    print("==================== KL Divergence for each state ===================")
    _p, _q, divergence = hlp.kl_divergence_for_each_state(p, q)

    print(pd.DataFrame(_p).to_string())
    print(pd.DataFrame(_q).to_string())
    print(pd.DataFrame(divergence).to_string())
    print(pd.DataFrame(_p).to_string())
    print(pd.DataFrame(_q).to_string())
    print(pd.DataFrame(divergence).to_string())

    print("================ Expected Leakage of the end state ================")
    future_dist = hlp.state_probability_after_n_steps(markov_chain, starting_state, 100)
    print("state probability after 100 steps")
    hlp.print_world(future_dist, shape=(3, 4))
    print("minimum non-zero probability")
    least_likely_future_state = np.where(future_dist == np.min(future_dist[np.nonzero(future_dist)]))
    print(least_likely_future_state[0][0])
    print(future_dist[least_likely_future_state])
    probabilities = []
    divergences = []
    for a in A:
        obs = a[0]
        probability = a[1]
        probabilities.append(probability)
        obs = [obs[i] + 1 for i in range(len(obs))]
        russelhmm = hmm.HMM(np.array(trans_p), np.array(emit_p), np.array(start_p))
        posterior_marginals = russelhmm.forward_backward(obs)
        p = posterior_marginals[1:]
        q = state_history[:len(obs)]
        _p, _q, divergence = hlp.kl_divergence_for_each_state(p, q)
        divergences.append(divergence[-1][end_state])
    print("probabilities accounted for")
    print(probabilities)
    print(sum(probabilities))
    print(divergences)
    expected_leakage = [divergences[i] * probabilities[i] for i in range(len(probabilities))]
    most_surprising_dist = [0] * len(states)
    most_surprising_dist[least_likely_future_state[0][0]] = 1.0
    print(most_surprising_dist)
    p = [most_surprising_dist]
    q = [future_dist.tolist()]
    _p, _q, most_surprising_divergence = hlp.kl_divergence_for_each_state(p, q)
    remaining_probability = 1 - sum(probabilities)
    remaining_possible_leakage = most_surprising_divergence[-1][end_state] * remaining_probability
    print(remaining_possible_leakage)
    print('Upper Bound:')
    print(sum(expected_leakage + [remaining_possible_leakage]))
    print('Lower Bound:')
    print(sum(expected_leakage))


def main():
    main_iterative()
    # main_linalg()


if __name__ == "__main__":
    main()
