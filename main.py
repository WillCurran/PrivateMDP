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
import policy_iteration as russel_norvig_world
import policy_iteration2 as river_world
import eppstein


def run_russel_norvig_world():
    """Run calculation under russel and norvig world.

    """
    hlp.print_h1('compute optimal policy')
    T, p, u, r, gamma = russel_norvig_world.main_iterative()
    hlp.print_h1("a priori analysis")

    hlp.print_h2('enumerate all policies')
    start_state = 8
    states = [i for i in range(12)]
    actions = [i for i in range(4)]
    start_p = [0.0 for i in range(12)]
    start_p[start_state] = 1.0

    policies = hlp.enumerate_policies(states, actions, [5], [3, 7])
    print("first policy returned:")
    russel_norvig_world.print_policy(policies[0], (3, 4))
    utility = russel_norvig_world.return_policy_evaluation(policies[0], u, r, T, gamma)
    hlp.print_world(utility)
    print("last policy returned:")
    russel_norvig_world.print_policy(policies[-1], (3, 4))
    utility = russel_norvig_world.return_policy_evaluation(policies[-1], u, r, T, gamma)
    hlp.print_world(utility)
    print("middle policy returned:")
    middleIndex = math.floor((len(policies) - 1) / 2)
    print(middleIndex)
    russel_norvig_world.print_policy(policies[middleIndex], (3, 4))
    utility = russel_norvig_world.return_policy_evaluation(policies[middleIndex], u, r, T, gamma)
    hlp.print_world(utility)
    print("bad policy:")
    middleIndex = math.floor((len(policies) - 1) / 2)
    bad_policy = (2, 2, 2, -1, 2, np.NaN, 2, -1, 2, 2, 2, 2)
    russel_norvig_world.print_policy(bad_policy, (3, 4))
    utility = russel_norvig_world.return_policy_evaluation(bad_policy, u, r, T, gamma)
    hlp.print_world(utility)


def run_river_world():
    """Run calculation under our custom river world.

    """
    # river_world.main_iterative()
        # Define an MDP


def run_russel_norvig_world_optimal_policy_viterbi_path_only():
    """Run calculation under russel and norvig world.

    This version only analyzes the viterbi path to the end state using the optimal policy and the most likely sequence of actions.
    """
    hlp.print_h1('create markov decision process and compute optimal policy')
    T, p, u, r, gamma = russel_norvig_world.main_iterative()
    hlp.print_h1("a priori analysis")
    hlp.print_h2("Create Markov Chain using MDP and Policy")
    print("optimal policy: ")
    policy = [np.NaN if np.isnan(i) else int(i) for i in p]
    russel_norvig_world.print_policy(policy, (3, 4))
    print("markov chain:")
    markov_chain = hlp.to_markov_chain(policy, T, 12)
    markov_chain_df = pd.DataFrame(markov_chain)
    print(markov_chain_df.to_string())
    # set obstacles to loop
    markov_chain[5][5] = 1.0
    # set terminal state to loop
    markov_chain[3][3] = 1.0
    markov_chain[7][7] = 1.0
    
    hlp.print_h2("create hidden markov model with mdp and policy")
    start_state = 8
    print('starting state')
    print(start_state)
    end_state = 3
    print('ending state')
    print(end_state)
    states, start_p, trans_p, emit_p = hlp.to_hidden_markov_model(T, p, 12, 4, start_state)
    print('states')
    hlp.print_world(states)
    print('starting distribution')
    hlp.print_world(start_p)
    print('transition distribution')
    hlp.print_table(trans_p)
    print('emissions distribution')
    hlp.print_table(emit_p)
    hlp.print_h2('compute most likely sequence of hidden states to the end state')
    print('states')
    D = (dk.dijkstra(trans_p, start_state, end_state, None, 3))
    print(D)
    print('probability')
    print(dk.path_prob(D, trans_p))
    hlp.print_h2('compute most likely sequence of actions to the end state')
    # A = dk.kdijkstra_actions(trans_p, start_state, end_state, 1, p, 1)
    state_index_list, A = eppstein.extract_data("russelworld.txt", p)
    print('result')
    print(A[0])
    obs = A[0][0]
    actions = [hlp.action_to_str_russel_norvig_world(a) for a in obs]
    print(actions)
    # obs needs positive indices for viterbi alg implementation below
    # obs_original = obs
    obs = [obs[i] + 1 for i in range(len(obs))]
    print(obs)
    hlp.print_h2('prior marginals')
    n_trans_p, n_trans_p_history = hlp.state_probabilities_up_to_n_steps(markov_chain, start_p, 100)
    n_trans_p = np.array(n_trans_p)
    print("state probability after 100 steps")
    hlp.print_world(n_trans_p)
    print("minimum non-zero probability")
    least_likely_future_state = np.where(n_trans_p == np.min(n_trans_p[np.nonzero(n_trans_p)]))
    print(least_likely_future_state[0][0])
    hlp.print_h2('posterior marginals')
    # Set obstacle states to loop
    trans_p[5][5] = 1.0
    # Set Terminal states to loop
    trans_p[7][7] = 1.0
    russelhmm = hmm.HMM(np.array(trans_p), np.array(emit_p), np.array(start_p))
    posterior_marginals = russelhmm.forward_backward(obs)
    print('forward backward result:')
    hlp.print_table(posterior_marginals)
    hlp.print_h2('difference between prior and posterior marginals')
    p = posterior_marginals[1:]
    q = n_trans_p_history[:len(obs)]
    rows = len(p)
    cols = len(p[0])
    for i in range(rows):
        print(kl_div(p[i], q[i]))
        print(sum(kl_div(p[i], q[i])))


def run_russel_norvig_world_old(obs=[]):
    hlp.print_h1('create markov decision process and compute optimal policy')
    T, p, u, r, gamma = russel_norvig_world.main_iterative()
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
    start_p = [0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               1.0, 0.0, 0.0, 0.0]
    state, state_history = hlp.state_probabilities_up_to_n_steps(markov_chain, start_p, 20)
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
    equilibrium_dist = hlp.equilibrium_distribution_power_iteration_3d_cols_left(T, np.array(start_p))
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
    # end_state = 3
    # D = (dk.dijkstra(trans_p, start_state, end_state, None, 3))
    # print(D)
    # print(dk.path_prob(D, trans_p))

    print("=========================== KDijkstra ===========================")

    # A = dk.kdijkstra_actions(trans_p, start_state, end_state, 10, p, 10)
    epp_states, epp_actions = eppstein.extract_data("russelworld.txt", p)
    print(epp_actions)

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
    hlp.print_world(states)
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
    future_dist = hlp.state_probability_after_n_steps(markov_chain, start_p, 100)
    print("state probability after 100 steps")
    hlp.print_world(future_dist)
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


def run_river_world_old(obs=[]):
    hlp.print_h1('create markov decision process and compute optimal policy')
    T, p, u, r, gamma = river_world.main_iterative()
    river_world.print_policy(p, shape=(3, 4))
    print("=======================A Priori Analysis========================")
    print("==================MDP + Policy = Markov Chain===================")
    print("Policy: ")
    print([int(i) for i in p])
    print("Markov Chain:")
    markov_chain = hlp.to_markov_chain([int(i) for i in p], T, 12)
    markov_chain_df = pd.DataFrame(markov_chain)
    print(markov_chain_df.to_string())

    # set terminal state
    markov_chain[7][7] = 1.0
    # set start state
    start_p = [0.0, 0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0]
    state, state_history = hlp.state_probabilities_up_to_n_steps(markov_chain, start_p, 20)
    print("Stationary Distribution:")
    print(state)
    state_history_df = pd.DataFrame(state_history)
    # state_history_df.plot()
    # plt.show()
    print(state_history_df.to_string())
    print("===========================Create HMM==========================")
    start_state = 4
    states, start_p, trans_p, emit_p = hlp.to_hidden_markov_model(T, p, 12, 2, start_state)
    print("============================Dijkstra===========================")
    # print(trans_p)

    # g = trans_to_graph(trans_p)
    # D = dijkstra(g,"v4","v7")
    # D = (dk.dijkstra(trans_p, start_state, 7, None, 3))
    # print(D)
    # print(dk.path_prob(D, trans_p))

    print("===========================KDijkstra===========================")

    # A = dk.kdijkstra_actions(trans_p, start_state, 7, 10, p, 2)

    # print(*A, sep="\n")

    print("=======================Expected Visits==========================")
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
    # obs = A[0][0]
    print(obs)

    s = "["
    for a in obs:
        s += hlp.action_to_str_river_world(a) + ", "
    if len(s) > 1:
        print(s[:-2] + "]")
    else:
        print("[]")
    # obs needs positive indices for viterbi alg implementation below
    # obs_original = obs
    obs = [obs[i] + 1 for i in range(len(obs))]
    print(obs)
    print("===========================Analyze HMM==========================")
    end_state = 7
    trans_p[end_state][end_state] = 1.0  # setting terminal state?

    trans_p_df = pd.DataFrame(trans_p)
    emit_p_df = pd.DataFrame(emit_p)
    print("##OBSERVATIONS##")
    print(obs)
    print("##STATES##")
    hlp.print_world(states)
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
    print("=========================== Forward Backward ==========================")
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
    for i in range(rows):
        print(entropy(p[i], q[i]))
    print('##RELATIVE ENTROPY##')
    for i in range(rows):
        print(rel_entr(p[i], q[i]))
        print(sum(rel_entr(p[i], q[i])))
    print('##KL DIVERGENCE##')
    for i in range(rows):
        print(kl_div(p[i], q[i]))
        print(sum(kl_div(p[i], q[i])))
    print("==================== KL Divergence for each state ===================")
    _p, _q, expected_excess_surprise = hlp.kl_divergence_for_each_state(p, q)

    print(pd.DataFrame(_p).to_string())
    print(pd.DataFrame(_q).to_string())
    print(pd.DataFrame(expected_excess_surprise).to_string())


def main():
    hlp.print_h1('markov decision process policy leakage calculation program')
    while True:
        print('please select an option:')
        print('1) russel and norvig world')
        print('2) river world')
        print('3) russel and norvig world with optimal policy/viterbi path only')
        print('4) russel and norvig world OLD')
        print('5) river world OLD')
        selection = input('enter your selection: ')

        if selection == '1':
            print('you selected option 1')
            run_russel_norvig_world()
            break
        elif selection == '2':
            print('you selected option 2')
            run_river_world()
            break
        elif selection == '3':
            print('you selected option 3')
            run_russel_norvig_world_optimal_policy_viterbi_path_only()
            break
        elif selection == '4':
            print('you selected option 4')
            run_russel_norvig_world_old()
            break
        elif selection == '5':
            print('you selected option 5')
            run_river_world_old()
            break
        else:
            print('Invalid selection. Please try again.\n')


if __name__ == "__main__":
    main()
