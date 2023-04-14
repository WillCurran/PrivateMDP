import ast
import itertools
import math
import os
import random
import statistics
import time

from scipy.special import rel_entr, kl_div
from scipy.stats import entropy
from tabulate import tabulate
import jpype

from helpers import print_table
import Forward_Backward_Algiorithm_wikipedia as fb
import Viterbi_Algorithm_wikipedia as vt
import dijkstras as dk
import eppstein
import helpers as hlp
import hmm as hmm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pareto as pareto
import policy_iteration as russel_norvig_world
import policy_iteration2 as river_world


def examine_russel_norvig_world(exponent=3):
    file_name = 'out/run_russel_norvig_world_all_policies_10000_executions.csv'
    df = pd.read_csv(file_name)
    df['start_state_utility'] = df['Utility'].str.split(',', expand=True)[8]
    print(df['start_state_utility'])
    df['delta'] = df['Upper Bound'] - df['Lower Bound']
    df_grouped = df.groupby(['start_state_utility'])
    # .to_csv('selected_policies.csv', index=False)
    print(df_grouped['delta'].min())
    file_name = 'out/examine_russel_norvig_world_all_policies.csv'

    if os.path.isfile(file_name):
        print(f"The file '{file_name}' exists.")
    else:
        print(f"The file '{file_name}' does not exist.")

    hlp.print_h1('compute optimal policy')
    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()
    # idx = df_grouped['delta'].idxmin()
    idx = df_grouped['Lower Bound'].idxmin()
    print(df.iloc[idx])
    result_list = []
    start_state_utilities_list = []
    lower_bounds_list = []
    upper_bounds_list = []
    for p in df.iloc[idx].iterrows():
        # Iterate over the range of exponents
        for e in range(exponent + 1):

            # Calculate n = 10^exponent
            n = math.pow(10, e)
            policy_str = p[1]['Policy']
            policy_list = list(
                map(lambda x: np.nan if x == 'nan' else int(x), policy_str[1:-1].split(', ')))
            # Call the function with n and get the result
            # utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only_with_random_sample_observations(T, policy_list, r, gamma, int(n))
            utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only(
                T, policy_list, r, gamma, int(n))

            # Append the result and other object properties to the result list
            result_list.append({
                'Policy': policy_list,
                'Utility': p[1]['Utility'],
                'start_state_utility': p[1]['start_state_utility'],
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound,
                'exponent': e
            })
            start_state_utilities_list.append(p[1]['start_state_utility'])
            lower_bounds_list.append(lower_bound)
            upper_bounds_list.append(upper_bound)

    print(len(result_list))
    pareto.pareto_front(start_state_utilities_list,
                        lower_bounds_list, upper_bounds_list, "Pareto")


def examine_russel_norvig_world_seperated(exponent=3):
    file_name = 'out/run_russel_norvig_world_all_policies_10000_executions.csv'
    df = pd.read_csv(file_name)
    df['start_state_utility'] = df['Utility'].str.split(',', expand=True)[8]
    print(df['start_state_utility'])
    df['delta'] = df['Upper Bound'] - df['Lower Bound']
    df_grouped = df.groupby(['start_state_utility'])
    # .to_csv('selected_policies.csv', index=False)
    file_name = 'out/examine_russel_norvig_world_all_policies.csv'

    if os.path.isfile(file_name):
        print(f"The file '{file_name}' exists.")
    else:
        print(f"The file '{file_name}' does not exist.")

    hlp.print_h1('compute optimal policy')
    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()
    # print(df_grouped['delta'].min())
    idx = df_grouped['Lower Bound'].idxmin()
    # print(df.iloc[idx])
    # for i, p in enumerate(df.iloc[idx].iterrows()):
    #    policy_str = p[1]['Policy']
    #    policy_list = list(map(lambda x: np.nan if x == 'nan' else int(x), policy_str[1:-1].split(', ')))
    #    russel_norvig_world.print_policy(policy_list, (3, 4))
    # return
    result_array = np.empty((len(idx), exponent + 1), dtype=object)
    for i, p in enumerate(df.iloc[idx].iterrows()):
        # Iterate over the range of exponents
        for j in range(exponent + 1):

            # Calculate n = 10^exponent
            n = math.pow(10, j)
            policy_str = p[1]['Policy']
            policy_list = list(
                map(lambda x: np.nan if x == 'nan' else int(x), policy_str[1:-1].split(', ')))
            # Call the function with n and get the result
            print(i)
            print(j)
            utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only_with_random_sample_observations(
                T, policy_list, r, gamma, int(n))
            # utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only(T, policy_list, r, gamma, int(n))

            # Append the result and other object properties to the result list

            result = {
                'Policy': policy_list,
                'Utility': p[1]['Utility'],
                'start_state_utility': p[1]['start_state_utility'],
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound,
                'exponent': j
            }

            result_array[i][j] = result
            # Create an empty list to hold the 'exponent' values
    # Iterate over the columns in the 2D array of objects
    for j in range(exponent + 1):
        column_objects = result_array[:, j]
        exponents_list = []
        start_state_utilities_list = []
        lower_bounds_list = []
        upper_bounds_list = []
        for obj in column_objects:
            exponents_list.append(obj['exponent'])
            start_state_utilities_list.append(obj['start_state_utility'])
            lower_bounds_list.append(obj['Lower Bound'])
            upper_bounds_list.append(obj['Upper Bound'])
        pareto.pareto_front(start_state_utilities_list, lower_bounds_list,
                            upper_bounds_list, "10^" + str(exponents_list[0]) + " policy executions")


def run_russel_norvig_world_optimal_policy_iteration_history(num_samples=1):
    """Run calculation under russel and norvig world.

    """
    hlp.print_h1('compute optimal policy')
    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()
    print(len(p_hist))
    hlp.print_h1("a priori analysis")

    print("Optimal Policy:")
    russel_norvig_world.print_policy(p, (3, 4))
    utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only_with_random_sample_observations(
        T, p, r, gamma, num_samples)
    print('*')
    hlp.print_world(utility)
    print(upper_bound)
    print(lower_bound)

    result = []
    for p in p_hist:
        print(p)
        result.append(run_russel_norvig_world_single_policy_only_with_random_sample_observations(
            T, p, r, gamma, num_samples))
        # russel_norvig_world.print_policy(p, (3, 4))
    print(len(result))
    print(result[0])
    print(result[-1])

    average_utility = []
    lowers = []
    uppers = []
    for r in result:
        # average_utility.append(np.sum(r[0]) / (len(r) - 1))
        average_utility.append(r[0][8])
        lowers.append(r[2])
        uppers.append(r[1])
    pareto.pareto_front(average_utility, lowers, uppers, "Pareto")


def run_russel_norvig_world_all_policies(num_samples=1, num_policies=np.inf):
    """Run calculation under russel and norvig world.

    """
    hlp.print_h1('compute optimal policy')
    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()

    hlp.print_h1("a priori analysis")

    print("Optimal Policy:")
    russel_norvig_world.print_policy(p, (3, 4))
    print(p)
    # utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only_with_random_sample_observations(T, p, r, gamma, num_samples)
    # print('*')
    # hlp.print_world(utility)
    # print(upper_bound)
    # print(lower_bound)

    hlp.print_h2('enumerate all policies')
    start_state = 8
    states = [i for i in range(12)]
    actions = [i for i in range(4)]
    start_p = [0.0 for i in range(12)]
    start_p[start_state] = 1.0

    policies = hlp.enumerate_policies(states, actions, [5], [3, 7])
    # result = []
    # for p in policies:
    #    result.append(run_russel_norvig_world_single_policy_only_with_random_sample_observations(T, p, r, gamma, num_samples))
    # print(len(result))

    file_name = 'out/run_russel_norvig_world_all_policies.csv'

    if os.path.isfile(file_name):
        print(f"The file '{file_name}' exists.")
        df = pd.read_csv(file_name)
    else:
        print(f"The file '{file_name}' does not exist.")
        length = int(np.minimum(len(policies), num_policies))
        state_utilities = np.empty((length, len(states)))
        start_state_utilities = np.empty(length)
        lower_bounds = np.empty(length)
        upper_bounds = np.empty(length)
        # average_utility.append(np.sum(r[0]) / (len(r) - 1))
        # lowers.append(r[2])
        # uppers.append(r[1])
        if(np.isinf(num_policies)):
            random.shuffle(policies)
        # for i in range(3):
            # p = list(random.choice(policies)) #Has possibility of repeating policies
            # p = policies[i]'
        for i in range(length):
            p = policies[i]
            result = run_russel_norvig_world_single_policy_only_with_random_sample_observations(T, p, r, gamma, num_samples)
            # result = run_russel_norvig_world_single_policy_only(T, p, r, gamma, num_samples)
            state_utilities[i,:] = result[0]
            start_state_utilities[i] = result[0][8]
            lower_bounds[i] = result[2]
            upper_bounds[i] = result[1]
        combined_list = [[policy, utility, lower, upper] for policy, utility, lower, upper in zip(
            policies, state_utilities, lower_bounds, upper_bounds)]
        df = pd.DataFrame(combined_list, columns=[
                          'Policy', 'Utility', 'Lower Bound', 'Upper Bound'])
        df['Utility'] = df['Utility'].apply(lambda x: str(list(x)))
        # df.to_csv(file_name, index=False)
    # extract the required lists
    # policy_list = df['Policy'].tolist()
    df['Utility'] = df['Utility'].apply(ast.literal_eval)
    start_state_utilities_list = df['Utility'].apply(
        lambda x: x[8] * -1).tolist()
    lower_bounds_list = df['Lower Bound'].tolist()
    upper_bounds_list = df['Upper Bound'].tolist()
    # pareto.plot_bounds(start_state_utilities_list, lower_bounds_list, upper_bounds_list, "Pareto")
    # pareto.pareto_front(start_state_utilities_list, lower_bounds_list, upper_bounds_list, "Pareto")
    # pareto.pareto_front_separate(start_state_utilities_list, lower_bounds_list, upper_bounds_list, "Lower Bound Pareto Front", "Upper Bound Pareto")


def run_russel_norvig_world_sample_policies(num_samples=1):
    """Run calculation under russel and norvig world.

    """
    hlp.print_h1('compute optimal policy')
    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()
    hlp.print_h1("a priori analysis")

    print("Optimal Policy:")
    russel_norvig_world.print_policy(p, (3, 4))
    utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only_with_random_sample_observations(
        T, p, r, gamma, num_samples)
    hlp.print_world(utility)
    print(upper_bound)
    print(lower_bound)

    hlp.print_h2('enumerate all policies')
    start_state = 8
    states = [i for i in range(12)]
    actions = [i for i in range(4)]
    start_p = [0.0 for i in range(12)]
    start_p[start_state] = 1.0

    policies = hlp.enumerate_policies(states, actions, [5], [3, 7])
    print("first policy returned:")
    russel_norvig_world.print_policy(policies[0], (3, 4))
    utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only_with_random_sample_observations(
        T, policies[0], r, gamma, num_samples)
    print('*')
    hlp.print_world(utility)
    print(upper_bound)
    print(lower_bound)
    print("last policy returned:")
    russel_norvig_world.print_policy(policies[-1], (3, 4))
    utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only_with_random_sample_observations(
        T, policies[-1], r, gamma, num_samples)
    hlp.print_world(utility)
    print(upper_bound)
    print(lower_bound)
    print("middle policy returned:")
    middleIndex = math.floor((len(policies) - 1) / 2)
    print(middleIndex)
    russel_norvig_world.print_policy(policies[middleIndex], (3, 4))
    utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only_with_random_sample_observations(
        T, policies[middleIndex], r, gamma, num_samples)
    hlp.print_world(utility)
    print(upper_bound)
    print(lower_bound)
    print("bad policy:")
    bad_policy = (2, 2, 2, -1, 2, np.NaN, 2, -1, 2, 2, 2, 2)
    russel_norvig_world.print_policy(bad_policy, (3, 4))
    utility, upper_bound, lower_bound = run_russel_norvig_world_single_policy_only_with_random_sample_observations(
        T, bad_policy, r, gamma, num_samples)
    hlp.print_world(utility)
    print(upper_bound)
    print(lower_bound)


def run_river_world():
    """Run calculation under our custom river world.

    """
    # river_world.main_iterative()
    # Define an MDP


def run_russel_norvig_world_single_policy_only(T, p, r, gamma, k):
    print('eppstein')
    """Run calculation under russel and norvig world.

    This version only a given policy and has no print statements

    Returns a tuple with upper and lower bound expected leakage
    """
    u = np.array([0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0])
    u = russel_norvig_world.return_policy_evaluation(p, u, r, T, gamma)
    # hlp.print_h1("a priori analysis")
    # hlp.print_h2("Create Markov Chain using MDP and Policy")
    # print("optimal policy: ")
    policy = [np.NaN if np.isnan(i) else int(i) for i in p]
    # russel_norvig_world.print_policy(policy, (3, 4))
    # print("markov chain:")
    markov_chain = hlp.to_markov_chain(policy, T, 12)
    markov_chain_df = pd.DataFrame(markov_chain)
    # print(markov_chain_df.to_string())
    # set obstacles to loop
    # markov_chain[5][5] = 1.0
    # set terminal state to loop
    # markov_chain[3][3] = 1.0
    # markov_chain[7][7] = 1.0

    # hlp.print_h2("create hidden markov model with mdp and policy")
    start_state = 8
    # print('starting state')
    # print(start_state)
    end_state = 3
    # print('ending state')
    # print(end_state)
    states, start_p, trans_p, emit_p = hlp.to_hidden_markov_model(
        T, p, 12, 4, start_state)
    # print('states')
    # hlp.print_world(states)
    # print('starting distribution')
    # hlp.print_world(start_p)
    # print('transition distribution')
    # hlp.print_table(trans_p)
    # print('emissions distribution')
    # hlp.print_table(emit_p)
    # hlp.print_h2('compute most likely sequence of hidden states to the end state')
    # print('states')
    future_dist, n_trans_p_history = hlp.state_probabilities_up_to_n_steps(
        markov_chain, start_p, 100)
    # future_dist = np.array(future_dist)
    # print("state probability after 100 steps")
    # hlp.print_world(future_dist)
    # print("minimum non-zero probability")
    least_likely_future_state = np.where(
        future_dist == np.min(future_dist[np.nonzero(future_dist)]))[0][0]
    # print(least_likely_future_state)
    # hlp.print_h2('posterior marginals')
    # Set obstacle states to loop
    # trans_p[5][5] = 1.0
    # Set Terminal states to loop
    # trans_p[7][7] = 1.0
    # trans_p[end_state][end_state] = 1.0
    russelhmm = hmm.HMM(np.array(trans_p), np.array(emit_p), np.array(start_p))

    # A = eppstein.eppstein(trans_p, p, start_state, end_state, k)

    # start_states = [0, 1, 2, 4, 6, 8, 9, 10, 11]
    start_states = [8]
    terminal_states = [3]

    A = []
    for start, terminal in itertools.product(start_states, terminal_states):
        # print(f"Start state: {start}, Terminal state: {terminal}")
        A += eppstein.eppstein(trans_p, p, start, terminal, k)

    # posterior_marginals = russelhmm.forward_backward(obs)
    # hlp.print_h2("expected leakage of the end state")
    # future_dist = hlp.state_probability_after_n_steps(markov_chain, start_p, 100)

    # print("state probability after 100 steps")
    # hlp.print_world(future_dist)
    # print("minimum non-zero state probability")
    length = len(A)
    # print(future_dist[least_likely_future_state])
    probabilities = np.zeros(length)
    divergences = np.zeros(length)
    for i, a in enumerate(A):
        obs = a[2]
        obs = [obs[i] + 1 for i in range(len(obs))]
        probabilities[i] = russelhmm.observation_prob(obs)  # a[0]
        russelhmm = hmm.HMM(np.array(trans_p), np.array(
            emit_p), np.array(start_p))
        posterior_marginals = russelhmm._forward_backward(obs).T
        # p = posterior_marginals[1:]
        # q = n_trans_p_history[:len(obs)]
        p = [posterior_marginals[-1]]
        # _forward_backward output does not include initial state probabilities, state_probabilities_up_to_n_steps does
        max_steps = np.minimum(len(n_trans_p_history) - 1, len(obs))
        q = [n_trans_p_history[max_steps]]
        _p, _q, divergence = hlp.kl_divergence_for_each_state(p, q)

        divergences[i] = sum(divergence[-1])
    # print("probabilities accounted for")
    # print(probabilities)
    # print(sum(probabilities))
    # print(len(divergences))
    # print(divergences)
    expected_leakage = divergences * probabilities
    print(expected_leakage)
    most_surprising_dist = np.zeros(len(states))
    most_surprising_dist[least_likely_future_state] = 1.0

    # print('most surprising state distribution')
    # print(most_surprising_dist)
    p = np.array([most_surprising_dist])
    q = np.array([future_dist])
    print(least_likely_future_state)
    print(p)
    print(q)
    _p, _q, most_surprising_divergence = hlp.kl_divergence_for_each_state(p, q)
    remaining_probability = 1 - sum(probabilities)
    print(remaining_probability)
    expected_kl_divergence = sum(most_surprising_divergence[-1])
    print(expected_kl_divergence)
    remaining_possible_leakage = expected_kl_divergence * remaining_probability
    print(remaining_possible_leakage)
    # print('remaining possible leakage')
    # print(remaining_possible_leakage)
    # print('Upper Bound:')
    # print(sum(expected_leakage + [remaining_possible_leakage]))
    # print('Lower Bound:')
    # print(sum(expected_leakage))
    return (u, sum(expected_leakage + [remaining_possible_leakage]), sum(expected_leakage))


def run_russel_norvig_world_single_policy_only_with_random_sample_observations(T, p, r, gamma, num_samples):
    print('random sampling')
    russel_norvig_world.print_policy(p, (3, 4))
    """Run calculation under russel and norvig world.

    This version only a given policy and has no print statements

    Returns a tuple with upper and lower bound expected leakage
    """
    u = np.array([0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0])
    u = russel_norvig_world.return_policy_evaluation(p, u, r, T, gamma)
    # hlp.print_h1("a priori analysis")
    # hlp.print_h2("Create Markov Chain using MDP and Policy")
    # print("optimal policy: ")
    policy = [np.NaN if np.isnan(i) else int(i) for i in p]
    # russel_norvig_world.print_policy(policy, (3, 4))
    # print("markov chain:")
    markov_chain = hlp.to_markov_chain(policy, T, 12)
    markov_chain_df = pd.DataFrame(markov_chain)
    # print(markov_chain_df.to_string())
    # set obstacles to loop
    # markov_chain[5][5] = 1.0
    # set terminal state to loop
    # markov_chain[3][3] = 1.0
    # markov_chain[7][7] = 1.0

    # hlp.print_h2("create hidden markov model with mdp and policy")
    start_state = 8
    # print('starting state')
    # print(start_state)
    end_state = 3
    # print('ending state')
    # print(end_state)
    states, start_p, trans_p, emit_p = hlp.to_hidden_markov_model(
        T, p, 12, 4, start_state)
    # print('states')
    # hlp.print_world(states)
    # print('starting distribution')
    # hlp.print_world(start_p)
    # print('transition distribution')
    # hlp.print_table(trans_p)
    # print('emissions distribution')
    # hlp.print_table(emit_p)
    # hlp.print_h2('compute most likely sequence of hidden states to the end state')
    # print('states')
    future_dist, n_trans_p_history = hlp.state_probabilities_up_to_n_steps(
        markov_chain, start_p, 100)
    # future_dist = np.array(future_dist)
    # print("state probability after 100 steps")
    # hlp.print_world(future_dist)
    # print("minimum non-zero probability")
    least_likely_future_state = np.where(
        future_dist == np.min(future_dist[np.nonzero(future_dist)]))[0][0]
    # print(least_likely_future_state)
    # hlp.print_h2('posterior marginals')
    # Set obstacle states to loop
    # trans_p[5][5] = 1.0
    # Set Terminal states to loop
    # trans_p[7][7] = 1.0
    # trans_p[end_state][end_state]

    russelhmm = hmm.HMM(np.array(trans_p), np.array(emit_p), np.array(start_p))

    A = []
    unique_obs = set()
    # while len(A) < num_samples:
    # print(num_samples)
    for i in range(num_samples):
        obs = hlp.execute_policy(p, T, start_state, 12)
        obs = [obs[i] + 1 for i in range(len(obs))]
        obs_tuple = tuple(obs)  # Convert list to tuple for set membership test
        if obs_tuple not in unique_obs:
            prob = russelhmm.observation_prob(obs)
            A.append((obs, prob))
            unique_obs.add(obs_tuple)
    # print(A)

    # posterior_marginals = russelhmm.forward_backward(obs)
    # hlp.print_h2("expected leakage of the end state")
    # future_dist = hlp.state_probability_after_n_steps(markov_chain, start_p, 100)
    # print("state probability after 100 steps")
    # hlp.print_world(future_dist)
    # print("minimum non-zero state probability")
    # least_likely_future_state = np.where(future_dist == np.min(future_dist[np.nonzero(future_dist)]))
    # print(least_likely_future_state)
    length = len(A)
    # print(future_dist[least_likely_future_state])
    probabilities = np.zeros(length)
    divergences = np.zeros(length)
    for i, a in enumerate(A):
        obs = a[0]
        probabilities[i] = a[1]
        actions = [hlp.action_to_str_russel_norvig_world(a) for a in obs]
        print(actions)
        # obs = [obs[i] + 1 for i in range(len(obs))]
        russelhmm = hmm.HMM(np.array(trans_p), np.array(
            emit_p), np.array(start_p))
        posterior_marginals = russelhmm._forward_backward(obs).T
        # p = posterior_marginals[1:]
        # q = n_trans_p_history[:len(obs)]
        p = [posterior_marginals[-1]]
        # _forward_backward output does not include initial state probabilities, state_probabilities_up_to_n_steps does
        max_steps = np.minimum(len(n_trans_p_history) - 1, len(obs))
        q = [n_trans_p_history[max_steps]]
        _p, _q, divergence = hlp.kl_divergence_for_each_state(p, q)
        divergences[i] = sum(divergence[-1])
    # print("probabilities accounted for")
    # print(probabilities)
    # print(sum(probabilities))
    # print(len(divergences))
    # print(divergences)
    expected_leakage = divergences * probabilities

    most_surprising_dist = np.zeros(len(states))
    most_surprising_dist[least_likely_future_state] = 1.0

    # print('most surprising state distribution')
    # print(most_surprising_dist)
    p = np.array([most_surprising_dist])
    q = np.array([future_dist])
    _p, _q, most_surprising_divergence = hlp.kl_divergence_for_each_state(p, q)
    remaining_probability = 1 - sum(probabilities)
    expected_kl_divergence = sum(most_surprising_divergence[-1])
    remaining_possible_leakage = expected_kl_divergence * remaining_probability

    print('p')
    print(p)
    print('q')
    print(q)
    print('expected kl divergence')
    print(expected_kl_divergence)
    print('total probability accounted for')
    print(len(probabilities))
    print(sum(probabilities))
    print('remaining probability')
    print(remaining_probability)
    print('remaining possible leakage')
    print(remaining_possible_leakage)
    print('Upper Bound:')
    print(sum(expected_leakage + [remaining_possible_leakage]))
    print('Lower Bound:')
    print(sum(expected_leakage))
    return (u, sum(expected_leakage + [remaining_possible_leakage]), sum(expected_leakage))


def run_russel_norvig_world_optimal_policy_only():
    """Run calculation under russel and norvig world.

    This version only analyzes the viterbi path to the end state using the optimal policy and the most likely sequence of actions.
    """
    hlp.print_h2('create markov decision process and compute optimal policy')
    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()
    hlp.print_h2("a priori analysis")
    hlp.print_h2("Create Markov Chain using MDP and Policy")
    print("optimal policy: ")
    policy = [np.NaN if np.isnan(i) else int(i) for i in p]
    russel_norvig_world.print_policy(policy, (3, 4))
    print("markov chain:")
    markov_chain = hlp.to_markov_chain(policy, T, 12)
    markov_chain_df = pd.DataFrame(markov_chain)
    print(markov_chain_df.to_string())
    # set obstacles to loop
    # markov_chain[5][5] = 1.0
    # set terminal state to loop
    # markov_chain[3][3] = 1.0
    # markov_chain[7][7] = 1.0

    hlp.print_h2("create hidden markov model with mdp and policy")
    start_state = 8
    print('starting state')
    print(start_state)
    end_state = 3
    print('ending state')
    print(end_state)
    states, start_p, trans_p, emit_p = hlp.to_hidden_markov_model(
        T, p, 12, 4, start_state)
    print('states')
    hlp.print_world(states)
    print('starting distribution')
    hlp.print_world(start_p)
    print('transition distribution')
    hlp.print_table(trans_p)
    print('emissions distribution')
    hlp.print_table(emit_p)

    hlp.print_h2('prior marginals')
    future_dist, n_trans_p_history = hlp.state_probabilities_up_to_n_steps(
        markov_chain, start_p, 100)
    future_dist = np.array(future_dist)
    print("state probability after 100 steps")
    hlp.print_world(future_dist)
    print("least likely future state")
    least_likely_future_state = np.where(
        future_dist == np.min(future_dist[np.nonzero(future_dist)]))[0][0]
    print(least_likely_future_state)

    hlp.print_h2('posterior marginals')
    # Set obstacle states to loop
    # trans_p[5][5] = 1.0
    # Set Terminal states to loop
    # trans_p[7][7] = 1.0
    # trans_p[end_state][end_state]
    russelhmm = hmm.HMM(np.array(trans_p), np.array(emit_p), np.array(start_p))

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

    hlp.print_h2(
        'compute most likely sequence of hidden states to the end state')
    V, prev = russelhmm.viterbi(obs)
    last_state = np.argmax(V[:, -1])
    path = list(russelhmm.build_viterbi_path(prev, last_state))[::-1]
    print(path)

    posterior_marginals = russelhmm.forward_backward(obs)
    hlp.print_h2("expected leakage of the end state")
    # future_dist = hlp.state_probability_after_n_steps(markov_chain, start_p, 100)
    print("state probability after 100 steps")
    hlp.print_world(future_dist)
    print("minimum non-zero state probability")
    print(future_dist[least_likely_future_state])
    probabilities = []
    divergences = []
    for a in A:
        obs = a[0]
        probability = a[1]
        probabilities.append(probability)
        obs = [obs[i] + 1 for i in range(len(obs))]
        russelhmm = hmm.HMM(np.array(trans_p), np.array(
            emit_p), np.array(start_p))
        posterior_marginals = russelhmm._forward_backward(obs).T
        p = posterior_marginals
        q = n_trans_p_history[1:len(obs) + 1]
        _p, _q, divergence = hlp.kl_divergence_for_each_state(p, q)
        divergences.append(sum(divergence[-1]))
    print("probabilities accounted for")
    # print(probabilities)
    print(sum(probabilities))
    print(len(divergences))
    # print(divergences)
    expected_leakage = [divergences[i] * probabilities[i]
                        for i in range(len(probabilities))]
    most_surprising_dist = np.zeros(len(states))
    most_surprising_dist[least_likely_future_state] = 1.0
    print('most surprising state distribution')
    print(most_surprising_dist)
    p = [most_surprising_dist]
    q = [future_dist.tolist()]
    _p, _q, most_surprising_divergence = hlp.kl_divergence_for_each_state(p, q)
    remaining_probability = 1 - sum(probabilities)
    expected_kl_divergence = sum(most_surprising_divergence[-1])
    remaining_possible_leakage = expected_kl_divergence * remaining_probability
    print('remaining possible leakage')
    print(remaining_possible_leakage)
    print('Upper Bound:')
    print(sum(expected_leakage + [remaining_possible_leakage]))
    print('Lower Bound:')
    print(sum(expected_leakage))


def run_russel_norvig_world_optimal_policy_viterbi_path_only():
    """Run calculation under russel and norvig world.

    This version only analyzes the viterbi path to the end state using the optimal policy and the most likely sequence of actions.
    """
    hlp.print_h1('create markov decision process and compute optimal policy')
    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()
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
    # markov_chain[5][5] = 1.0
    # set terminal state to loop
    # markov_chain[3][3] = 1.0
    # markov_chain[7][7] = 1.0

    hlp.print_h2("create hidden markov model with mdp and policy")
    start_state = 8
    print('starting state')
    print(start_state)
    end_state = 3
    print('ending state')
    print(end_state)
    states, start_p, trans_p, emit_p = hlp.to_hidden_markov_model(
        T, p, 12, 4, start_state)
    print('states')
    hlp.print_world(states)
    print('starting distribution')
    hlp.print_world(start_p)
    print('transition distribution')
    hlp.print_table(trans_p)
    print('emissions distribution')
    hlp.print_table(emit_p)

    hlp.print_h2('prior marginals')
    future_dist, n_trans_p_history = hlp.state_probabilities_up_to_n_steps(
        markov_chain, start_p, 100)
    future_dist = np.array(future_dist)
    print("state probability after 100 steps")
    hlp.print_world(future_dist)
    print("minimum non-zero state probability")
    least_likely_future_state = np.where(
        future_dist == np.min(future_dist[np.nonzero(future_dist)]))[0][0]
    print(least_likely_future_state)
    hlp.print_h2('posterior marginals')
    # Set obstacle states to loop
    # trans_p[5][5] = 1.0
    # Set Terminal states to loop
    # trans_p[7][7] = 1.0
    # trans_p[end_state][end_state]
    russelhmm = hmm.HMM(np.array(trans_p), np.array(emit_p), np.array(start_p))

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

    hlp.print_h2(
        'compute most likely sequence of hidden states to the end state')
    V, prev = russelhmm.viterbi(obs)
    last_state = np.argmax(V[:, -1])
    path = list(russelhmm.build_viterbi_path(prev, last_state))[::-1]
    print(path)

    posterior_marginals = russelhmm._forward_backward(obs).T
    print('forward backward result:')
    hlp.print_table(posterior_marginals)
    hlp.print_h2('difference between prior and posterior marginals')
    p = posterior_marginals
    q = n_trans_p_history[1:len(obs) + 1]

    rows = len(p)
    cols = len(p[0])
    for i in range(rows):
        print(kl_div(p[i], q[i]))
        print(sum(kl_div(p[i], q[i])))
    hlp.print_h2("KL divergence for each state")
    _p, _q, divergence = hlp.kl_divergence_for_each_state(p, q)

    print('p')
    hlp.print_table(p)
    print('q')
    hlp.print_table(q)
    print('_p')
    hlp.print_table(_p.tolist())
    print('_q')
    hlp.print_table(_q.tolist())
    print('divergence')
    hlp.print_table(divergence)


def run_russel_norvig_world_old(obs=[]):
    hlp.print_h1('create markov decision process and compute optimal policy')
    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()
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
    # markov_chain[5][5] = 1.0
    # set terminal state to loop
    # markov_chain[3][3] = 1.0
    # markov_chain[7][7] = 1.0
    # set start state
    start_p = [0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               1.0, 0.0, 0.0, 0.0]
    state, state_history = hlp.state_probabilities_up_to_n_steps(
        markov_chain, start_p, 20)
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
    # equilibrium = hlp.equilibrium_distribution(T)
    equilibrium_dist = hlp.equilibrium_distribution_power_iteration_3d_cols_left(
        T, np.array(start_p))
    print(equilibrium_dist)
    print("======================= Stationary Distribtuion of Markov Chain=======================")
    stationary_dist = hlp.stationary_distribution(np.array(markov_chain))
    print(stationary_dist)
    print("======================= Posteriori Analysis =====================")
    print("=========================== Create HMM ==========================")
    start_state = 8
    states, start_p, trans_p, emit_p = hlp.to_hidden_markov_model(
        T, p, 12, 4, start_state)
    # print("============================ Dijkstra ===========================")
    # print(trans_p)
    # g = trans_to_graph(trans_p)
    # D = dijkstra(g,"v4","v7")
    # end_state = 3
    # D = (dk.dijkstra(trans_p, start_state, end_state, None, 3))
    # print(D)
    # print(dk.path_prob(D, trans_p))
    # print("=========================== KDijkstra ===========================")
    # Viterbi Path
    # A = dk.kdijkstra_actions(trans_p, start_state, end_state, 1, p, 1)
    # print('states')
    # D = (dk.dijkstra(trans_p, start_state, end_state, None, 3))
    # print(D)
    # print('probability')
    # print(dk.path_prob(D, trans_p))
    # A = dk.kdijkstra_actions(trans_p, start_state, end_state, 10, p, 10)

    print("============================ Eppstein ============================")
    epp_states, epp_actions = eppstein.extract_data("russelworld.txt", p)
    print(epp_actions)
    A = epp_actions
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
    # trans_p[5][5] = 1.0
    # Set Terminal states to loop
    # trans_p[7][7] = 1.0
    end_state = 3
    # trans_p[end_state][end_state] = 1.0
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
    forward = russelhmm.forward(obs)
    backward = russelhmm.backward(obs)
    print(forward)
    print(backward)
    posterior_marginals = russelhmm._forward_backward(obs).T
    print("posterior marginals:")
    print(pd.DataFrame(posterior_marginals).to_string())
    print("========== Difference Between Prior and Posterior Marginals  ==========")
    p = posterior_marginals  # [1:]
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

    print('p')
    hlp.print_table(p)
    print('q')
    hlp.print_table(q)
    print('_p')
    hlp.print_table(_p.tolist())
    print('_q')
    hlp.print_table(_q.tolist())
    print('divergence')
    hlp.print_table(divergence)

    print("================ Expected Leakage of the end state ================")
    future_dist = hlp.state_probability_after_n_steps(
        markov_chain, start_p, 100)
    print("state probability after 100 steps")
    hlp.print_world(future_dist)
    print("minimum non-zero probability")
    least_likely_future_state = np.where(
        future_dist == np.min(future_dist[np.nonzero(future_dist)]))[0][0]
    print(least_likely_future_state)
    print(future_dist[least_likely_future_state])
    probabilities = []
    divergences = []
    for a in A:
        obs = a[0]
        probability = a[1]
        probabilities.append(probability)
        obs = [obs[i] + 1 for i in range(len(obs))]
        russelhmm = hmm.HMM(np.array(trans_p), np.array(
            emit_p), np.array(start_p))
        posterior_marginals = russelhmm._forward_backward(obs).T
        p = posterior_marginals  # [1:]
        q = state_history[:len(obs)]
        _p, _q, divergence = hlp.kl_divergence_for_each_state(p, q)
        divergences.append(divergence[-1][end_state])
    print("probabilities accounted for")
    print(probabilities)
    print(sum(probabilities))
    print(divergences)
    expected_leakage = [divergences[i] * probabilities[i]
                        for i in range(len(probabilities))]
    most_surprising_dist = np.zeros(len(states))
    most_surprising_dist[least_likely_future_state] = 1.0
    print(most_surprising_dist)
    p = [most_surprising_dist]
    q = [future_dist.tolist()]
    _p, _q, most_surprising_divergence = hlp.kl_divergence_for_each_state(p, q)
    remaining_probability = 1 - sum(probabilities)
    remaining_possible_leakage = most_surprising_divergence[-1][end_state] * \
        remaining_probability
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
    # markov_chain[7][7] = 1.0
    # set start state
    start_p = [0.0, 0.0, 0.0, 0.0,
               1.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0]
    state, state_history = hlp.state_probabilities_up_to_n_steps(
        markov_chain, start_p, 20)
    print("Stationary Distribution:")
    print(state)
    state_history_df = pd.DataFrame(state_history)
    # state_history_df.plot()
    # plt.show()
    print(state_history_df.to_string())
    print("===========================Create HMM==========================")
    start_state = 4
    states, start_p, trans_p, emit_p = hlp.to_hidden_markov_model(
        T, p, 12, 2, start_state)
    # print("============================Dijkstra===========================")
    # print(trans_p)

    # g = trans_to_graph(trans_p)
    # D = dijkstra(g,"v4","v7")
    # D = (dk.dijkstra(trans_p, start_state, 7, None, 3))
    # print(D)
    # print(dk.path_prob(D, trans_p))

    # print("===========================KDijkstra===========================")

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
    # trans_p[end_state][end_state] = 1.0  # setting terminal state?

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
    forward = russelhmm.forward(obs)
    backward = russelhmm.backward(obs)
    print(forward)
    print(backward)
    posterior_marginals = russelhmm._forward_backward(obs).T
    print("posterior marginals:")
    print(pd.DataFrame(posterior_marginals).to_string())
    print("========== Difference Between Prior and Posterior Marginals  ==========")
    p = posterior_marginals  # [1:]
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
    _p, _q, divergence = hlp.kl_divergence_for_each_state(p, q)

    print('p')
    hlp.print_table(p)
    print('q')
    hlp.print_table(q)
    print('_p')
    hlp.print_table(_p.tolist())
    print('_q')
    hlp.print_table(_q.tolist())
    print('divergence')
    hlp.print_table(divergence)


def main():
    jpype.startJVM()
    hlp.print_h1('markov decision process policy leakage calculation program')
    while True:
        print('please select an option:')
        print('1) russel and norvig world')
        print('2) river world')
        print('3) russel and norvig world with optimal policy only')
        print('4) russel and norvig world with optimal policy and viterbi path only')
        print('5) russel and norvig world OLD')
        print('6) river world OLD')
        print('7) examine russel and norvig world')
        selection = input('enter your selection: ')
        start_time = time.time()
        if selection == '1':
            print('you selected option 1')
            run_russel_norvig_world_all_policies(1, 1)
            # run_russel_norvig_world_sample_policies(10)
            break
        elif selection == '2':
            print('you selected option 2')
            run_river_world()
            break
        elif selection == '3':
            print('you selected option 3')
            run_russel_norvig_world_optimal_policy_only()
            break
        elif selection == '4':
            print('you selected option 4')
            run_russel_norvig_world_optimal_policy_viterbi_path_only()
            break
        elif selection == '5':
            print('you selected option 5')
            run_russel_norvig_world_old()
            break
        elif selection == '6':
            print('you selected option 6')
            run_river_world_old()
            break
        elif selection == '7':
            print('you selected option 7')
            examine_russel_norvig_world_seperated(0)
            break
        else:
            print('Invalid selection. Please try again.\n')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    jpype.shutdownJVM()


if __name__ == "__main__":
    main()
