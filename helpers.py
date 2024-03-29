import itertools
import math
import random
import sys

from header_printer import print_header
from scipy.special import rel_entr, kl_div
from scipy.stats import entropy
from tabulate import tabulate

import numpy as np
import pandas as pd

CMP_DELTA = 0.000001


def information_gain(Q, P, s, path_probability):
    """Calculate a bound for expected(?) information gain given one execution of viterbi on one string of observations
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
    worst_expected_entropy = path_probability * rel_entropy + \
        (1.0 - path_probability) * max_remaining_info
    return (known_rel_entropy, worst_expected_entropy)


def action_to_str_russel_norvig_world(a):
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


def action_to_str_river_world(a):
    result = '#'
    if a == -1:
        result = "DONE"
    elif a == 0:
        result = ">"
    elif a == 1:
        result = "|"
    return result

# def take_action(curr_state, action, T):
#     """Return the next state and the given current state and the action chosen
#
#     """
#     coin = random.random()
#     # coin = 0.5
#     # 12 possible next states
#     next_states = T[curr_state,:, int(action)]
#     prob_counter = 0.0
#     # randomly take next action based on weights
#     for state, prob in enumerate(next_states):
#         if coin < prob_counter + prob:
#             return state
#         prob_counter += prob
#     return -1

# def execute_policy(p, T, start, max_t):
#     """Place an agent in the environment and generate a stream of actions
#
#     """
#     curr_state = start
#     output = []
#     # no longer than max_t steps
#     for i in range(max_t):
#         output.append(int(p[curr_state]))
#         if p[curr_state] == -1:
#             break
#         curr_state = take_action(curr_state, p[curr_state], T)
#     return output


def take_action(state, action, T):
    """
    Given a state, an action, and a transition model T,
    return the next state based on the probabilities of the transition model.
    """
    next_states = np.nonzero(T[state,:, action])[0]
    probs = T[state, next_states, action]
    next_state = np.random.choice(next_states, p=probs)
    return next_state


def execute_policy(p, T, start, max_t):
    """Place an agent in the environment and generate a stream of actions."""
    curr_state = start
    output = np.empty(max_t, dtype=int)
    for i in range(max_t):
        action = int(p[curr_state])
        output[i] = action
        if action == -1:
            break
        curr_state = take_action(curr_state, action, T)
    return output[:i + 1]


def to_markov_chain(p, T, max_t):
    result = [[0] * max_t] * max_t
    for t in range(max_t):
        if not np.isnan(p[t]):
            result[t] = [row[p[t]] for row in T[t][:]]
    return result


def n_step_transition_matrix(T, n):
    # Initialize a list to store the matrix exponentiations
    result = []
    # Compute matrix exponentiations
    for i in range(n):
        # i+1 since the range starts from 0
        P_i = np.linalg.matrix_power(T, i + 1)
        result.append(P_i)
    return result


def state_probability_after_n_steps(T, start_state, n):
    """Calculates the n step transition matrix probability distribution
    to caclulate the state probability after n steps.

    """
    # Compute matrix exponentiation T^n
    P_n = np.linalg.matrix_power(T, n)

    # Compute the final state distribution after n steps
    final_dist = np.dot(start_state, P_n)
    return final_dist

# def state_probabilities_up_to_n_steps(markov_chain, start_p, power):
#     """Calculates the state probability after steps from 1 to n.
#     power is n or the number to raise the markov chain for to calculate
#     the n step transition matrix probability distribution.
#
#     """
#     state = [start_p]
#     state_history = [state[0]]
#     for x in range(power):
#         next_state = [[sum(a * b for a, b in zip(state_row, markov_chain_col))
#                        for markov_chain_col in zip(*markov_chain)]
#                       for state_row in state]
#         state_history.append(next_state[0])
#         state = next_state
#     return (state, state_history)


def state_probabilities_up_to_n_steps(markov_chain, start_p, power):
    """Calculates the state probability after steps from 1 to n.
    power is n or the number to raise the markov chain for to calculate 
    the n step transition matrix probability distribution.

    """
    state = np.array(start_p)
    state_history = np.empty((power + 1, len(start_p)))
    state_history[0] = start_p
    for x in range(power):
        next_state = np.dot(state, markov_chain)
        state_history[x + 1] = next_state
        state = next_state
    return (state, state_history)


def to_hidden_markov_model(transition_matrix, policy, number_of_states, number_of_observable_actions, start_state):
    states = [i for i in range(number_of_states)]
    start_p = [0.0 for i in range(number_of_states)]
    start_p[start_state] = 1.0

    # Viterbi needs 12x12 transition matrix
    # Generate the one induced by the policy

    # trans_p = []
    # for i in range(number_of_states):
    #     trans_p.append([0.0 for j in range(number_of_states)])
    #     if not np.isnan(policy[i]) and not policy[i] == -1:
    #         for j in range(number_of_states):
    #             trans_p[i][j] = transition_matrix[i, j, int(policy[i])]
    # # emmission probabilities are induced by the policy
    # emit_p = []
    # for i in range(number_of_states):
    #     emit_p.append([0.0 for j in range(number_of_observable_actions + 1)])
    #     # TODO - make nondeterministic policy possible
    #     if not np.isnan(policy[i]):
    #         # Increment observable actions by 1, index 0 means termination
    #         emit_p[i][int(policy[i]) + 1] = 1.00
    trans_p = np.zeros((number_of_states, number_of_states))
    for i in range(number_of_states):
        if not np.isnan(policy[i]) and not policy[i] == -1:
            for j in range(number_of_states):
                trans_p[i, j] = transition_matrix[i, j, int(policy[i])]

    emit_p = np.zeros((number_of_states, number_of_observable_actions + 1))
    for i in range(number_of_states):
        if not np.isnan(policy[i]):
            emit_p[i, int(policy[i]) + 1] = 1.0

    return (states, start_p, trans_p, emit_p)

# def kl_divergence_for_each_state(p, q):
#     rows = len(p)
#     cols = len(p[0])
#     _p = [[0] * cols for i in range(rows)]
#     _q = [[0] * cols for i in range(rows)]
#     result = [[0] * cols for i in range(rows)]
#     for i in range(rows):
#         for j in range(cols):
#             p_i_j = p[i][j]
#             q_i_j = q[i][j]
#
#             if p_i_j > 1.0:
#                 p_i_j = 1.0
#
#             if q_i_j > 1.0:
#                 q_i_j = 1.0
#
#             # compute Bernoulli distribution
#             _p[i][j] = [p_i_j, 1.0 - p_i_j]
#             _q[i][j] = [q_i_j, 1.0 - q_i_j]
#             result[i][j] = sum(kl_div(_p[i][j], _q[i][j]))
#     return (_p, _q, result)


def kl_divergence_for_each_state(p, q):
    p = np.clip(p, a_min=0.0, a_max=1.0)  # clip values above 1.0
    q = np.clip(q, a_min=0.0, a_max=1.0)  # clip values above 1.0

    # compute Bernoulli distribution
    _p = np.stack((p, 1 - p), axis=-1)
    _q = np.stack((q, 1 - q), axis=-1)

    # calculate KL divergence for each state
    # result = np.sum(_p * np.log(_p / _q), axis=-1)
    result = np.sum(kl_div(_p, _q), axis=-1)

    return (_p, _q, result)


def stationary_distribution(P):
    # Identify all-zero rows in the transition matrix
    zero_rows = np.where(~P.any(axis=1))[0]

    # Set the diagonal elements of the terminal states and obstacles to 1
    P[zero_rows, zero_rows] = 1

    # Find the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find the index of the eigenvalue of 1
    index = np.where(np.isclose(eigenvalues, 1))[0][0]

    # Normalize the eigenvector to obtain the stationary distribution
    stationary_dist = np.real(
        eigenvectors[:, index] / eigenvectors[:, index].sum())
    return stationary_dist


def equilibrium_distribution(transition_matrix, max_iter=100, tol=1e-8):
    """
    Calculates the equilibrium distribution using power iteration.
    transition_matrix: an array with shape (num_states, num_states, num_actions)
    max_iter: maximum number of iterations
    tol: tolerance for convergence
    """
    num_states, _, num_actions = transition_matrix.shape
    # Initialize distribution as uniform
    distribution = np.ones((num_states,)) / num_states
    for i in range(max_iter):
        prev_distribution = distribution
        # Calculate new distribution
        distribution = np.sum(
            transition_matrix[:,:, i] @ distribution, axis=1)
        # Check for convergence
        if np.linalg.norm(distribution - prev_distribution) < tol:
            break
    return distribution


def equilibrium_distribution_power_iteration_3d_cols_left(P, pi, epsilon=1e-8, max_iter=1000):
    """
    Calculate the equilibrium distribution for a 3-dimensional transition matrix indexed by source state, next state, and action and are stochastic by columns using power iteration method
    :param P: Transition matrix, indexed by source state, next state, and action
    :param pi: Initial distribution
    :param epsilon: Tolerance for convergence
    :param max_iter: Maximum number of iterations
    :return: Equilibrium distribution
    EX: 
    # Transition matrix, indexed by source state, next state, and action
    P = np.random.rand(12, 12, 4)

    # Initial distribution
    pi = np.ones(12)/12

    # Normalize the transition matrix by columns
    P /= P.sum(axis=-1, keepdims=True)

    # Calculate equilibrium distribution
    pi_star = equilibrium_distribution_power_iteration_3d_cols(P, pi)
    print(pi_star)
    """
    # Sum over the action dimension
    P_flat = P.sum(axis=-1)
    pi_star = pi.copy()
    # check if the matrix is irreducible and aperiodic
    if (np.linalg.matrix_power(P_flat, 100) > 0).all():
        # check if the initial distribution is a left eigenvector of the transition matrix
        pi_star /= pi_star.dot(P_flat)
        for _ in range(max_iter):
            pi_prev = pi_star.copy()
            pi_star = pi_star.dot(P_flat)
            if np.allclose(pi_prev, pi_star, atol=epsilon):
                break
    return pi_star


def is_irreducible_aperiodic(P):
    """
    Check if a matrix is irreducible and aperiodic
    :param P: Transition matrix
    :return: True if the matrix is irreducible and aperiodic, False otherwise
    EX: 
    # Transition matrix, indexed by source state, next state, and action
    P = np.random.rand(12, 12, 4)

    # Normalize the transition matrix by columns
    P /= P.sum(axis=-1, keepdims=True)

    if is_irreducible_aperiodic(P.sum(axis=-1)):
        pi_star = equilibrium_distribution_power_iteration_3d_cols(P, pi)
        print(pi_star)
    else:
        print("The matrix is not irreducible and aperiodic")

    """
    P_n = np.linalg.matrix_power(P, 100)
    return (P_n > 0).all()


def enumerate_policies(states, actions, obstacles, terminals):
    """enumerate all policies of an MDP

    defining the MDP as a tuple (S, A, T, R, gamma) where:

    S is the set of states
    A is the set of actions
    T is the transition model, which is a probability distribution over 
    the next state given the current state and action: T(s' | s, a)
    R is the reward function, which maps states and actions to real 
    values: R(s, a)
    gamma is the discount factor, which determines the importance of future 
    rewards compared to current rewards

    Then, you can define a function that takes an MDP as input and returns a 
    list of policies. A policy is a function that maps states to actions. 
    To enumerate all policies, you can create a list of all possible functions
    that map states to actions. For example, you can do this by creating a 
    list of all possible combinations of states and actions, and then creating 
    a function for each combination that returns the action for the corresponding 
    state.

    EX:
    # Define an MDP
    mdp = (['s1', 's2', 's3'], ['a1', 'a2'], T, R, gamma)

    # Enumerate the policies of the MDP
    policies = enumerate_policies(mdp)
    """
    number_of_policies = len(actions) ** (len(states) - 
                                          len(obstacles) - len(terminals))
    # Create the 2-d array of possible selections
    selections = [actions for i in range(len(states))]
    for obstacle in obstacles:
        selections[obstacle] = [np.NaN]
    for terminal in terminals:
        selections[terminal] = [-1]
    # Use itertools.product to generate all possible combinations of selections

    combinations = itertools.product(*selections)
    policies = np.fromiter(combinations, dtype=object)
    print("expected number of policies:")
    print(number_of_policies)
    print("actual number of policies")
    print(len(policies))
    return policies


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
    print("time=%d : %s" % (0, ', '.join(
        ["%.2f" % curr_p[st] for st in states]) + ": sum=%.2f" % sum(curr_p)))
    for i in range(1, t + 1):
        next_p = [0.0 for j in range(12)]
        for st in states:
            for next_st in states:
                next_p[next_st] += curr_p[st] * trans_p[st][next_st]
        for st in states:
            curr_p[st] = next_p[st]
        print("time=%d : %s" % (i, ', '.join(
            ["%.2f" % curr_p[st] for st in states]) + ": sum=%.2f" % sum(curr_p)))
    return curr_p


def print_world(arr, shape=(3, 4)):
    table = np.reshape(arr, shape)
    headers = np.arange(shape[1]) + 1
    df = pd.DataFrame(table)
    row_labels = np.flip(np.arange(shape[0])) + 1
    df.index = row_labels

    print(tabulate(df, headers=headers))


def print_h1(message):
    print_header(message.title())


def print_h2(message):
    print((" " + message + " ").center(80, '='))


def print_table(table):
    df = pd.DataFrame(table)
    print(df.to_string())


def kl_divergence_example():
    p = [9 / 25, 12 / 25, 4 / 25, 0, 0]
    q = [1 / 3, 1 / 3, 1 / 3, 0, 0]

    result_1 = np.sum(kl_div(p, q), axis=-1)
    result_2 = np.sum(kl_div(q, p), axis=-1)

    result_3 = kl_divergence_for_each_state(p, q)
    result_4 = kl_divergence_for_each_state(q, p)

    return (result_1, result_2, np.sum(result_3[2]), np.sum(result_4[2]))


def can_reach_terminal_states(start_state, terminal_states, T):
    # Get the number of states and actions from the transition model
    num_states, _, num_actions = T.shape

    # Initialize the set of reachable states with the start state
    reachable_states = {start_state}

    # Keep track of previously reachable states to detect convergence
    prev_reachable_states = set()

    # Repeat until convergence
    while reachable_states != prev_reachable_states:
        # Update the set of previously reachable states
        prev_reachable_states = reachable_states.copy()

        # For each reachable state, check if it can reach any new states
        for state in prev_reachable_states:
            for action in range(num_actions):
                for next_state in range(num_states):
                    if T[state, next_state, action] > 0:
                        reachable_states.add(next_state)
    return any(state in terminal_states for state in reachable_states)


def can_any_state_reach_terminal_states(T, terminal_states):
    n_states = T.shape[0]
    Q = np.zeros((n_states, n_states))
    for a in range(T.shape[2]):
        Q += T[:,:, a]
    for t in terminal_states:
        if np.any(Q[:, t] > 0):
            return True
    return False


def main():
    print("Calling main function in helpers.py")
    print(kl_divergence_example())
    # > > V +1 
    # ^ X V -1
    # > > V V
    # p = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0, 0.05, 0.05, .50, 0.05, 0.05, 0.05])
    p = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    q = np.array([4.86092583e-037, 7.16607983e-036, 1.31116421e-035, 2.99876161e-036,
                  2.04905117e-037, 0.00000000e+000, 3.11038308e-035, 7.11373796e-036,
                  1.00000000e-100, 5.74517874e-002, 4.59614299e-001, 4.59614297e-001])
    p = q
    result = kl_divergence_for_each_state(p, q)[2]
    print(result)
    print(sum(result))

    print(np.finfo(np.float64).min)  # smallest float64 number
    print(np.finfo(np.float64).max)  # largest float64 number
    print(np.finfo(np.float64).eps)  # represents the smallest positive floating-point number greater than 0 that can be represented on your system.
    print(sys.float_info.epsilon)
    print(1 - np.finfo(np.float64).eps)


if __name__ == "__main__":
    main()
