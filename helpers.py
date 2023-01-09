import itertools
import math
import random

from scipy.special import rel_entr, kl_div
from scipy.stats import entropy
from tabulate import tabulate

import numpy as np
import pandas as pd

CMP_DELTA = 0.000001


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


def take_action(curr_state, action, T):
    """Return the next state and the given current state and the action chosen

    """
    coin = random.random()
    # coin = 0.5
    # 12 possible next states
    next_states = T[curr_state,:, int(action)]
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


def to_markov_chain(p, T, max_t):
    result = [[0] * max_t] * max_t
    for t in range(max_t):
        if not np.isnan(p[t]):
            result[t] = [row[p[t]] for row in T[t][:]]
    return result


def stationary_distribution(markov_chain, starting_state, power):
    state = [starting_state]
    state_history = [state[0]]
    for x in range(power):
        next_state = [[sum(a * b for a, b in zip(state_row, markov_chain_col))
                       for markov_chain_col in zip(*markov_chain)]
                      for state_row in state]
        state_history.append(next_state[0])
        state = next_state
    return (state, state_history)


def to_hidden_markov_model(transition_matrix, policy, number_of_states, number_of_observable_actions, start_state):
    states = [i for i in range(number_of_states)]
    start_p = [0.0 for i in range(number_of_states)]
    start_p[start_state] = 1.0

    # Viterbi needs 12x12 transition matrix
    # Generate the one induced by the policy
    trans_p = []
    for i in range(number_of_states):
        trans_p.append([0.0 for j in range(number_of_states)])
        if not np.isnan(policy[i]) and not policy[i] == -1:
            for j in range(number_of_states):
                trans_p[i][j] = transition_matrix[i, j, int(policy[i])]
    # emmission probabilities are induced by the policy
    emit_p = []
    for i in range(number_of_states):
        emit_p.append([0.0 for j in range(number_of_observable_actions + 1)])
        # TODO - make nondeterministic policy possible
        if not np.isnan(policy[i]):
            # Increment observable actions by 1, index 0 means termination
            emit_p[i][int(policy[i]) + 1] = 1.0
    return (states, start_p, trans_p, emit_p)


def kl_divergence_for_each_state(p, q):
    rows = len(p)
    cols = len(p[0])
    _p = [[0] * cols for i in range(rows)]
    _q = [[0] * cols for i in range(rows)]
    result = [[0] * cols for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            p_i_j = p[i][j]
            q_i_j = q[i][j]

            if p_i_j > 1.0:
                p_i_j = 1.0

            if q_i_j > 1.0:
                q_i_j = 1.0

            _p[i][j] = [p_i_j, 1.0 - p_i_j]
            _q[i][j] = [q_i_j, 1.0 - q_i_j]
            result[i][j] = sum(kl_div(_p[i][j], _q[i][j]))
    return (_p, _q, result)


def equilibrium_distribution(transition_matrix):
    """Calculate the equlibrium distribution of a transition matrix

    This should give you the equilibrium distribution of the transition 
    matrix. It is important to note that the equilibrium distribution 
    only exists if the transition matrix is aperiodic and irreducible, 
    which means that it is possible to reach any state from any other 
    state in a finite number of steps and that there is no subset of states 
    that cannot be reached from any other state.
    """
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)
    index = np.where(np.isclose(eigenvalues, 1))[0][0]
    result = np.conj(eigenvectors[:, index]).T
    result = result / result.sum()
    return result


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
    number_of_policies = len(actions) ** (len(states) - len(obstacles) - len(terminals))
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


def print_world(arr, shape):
    table = np.reshape(arr, shape)
    headers = np.arange(shape[1]) + 1
    df = pd.DataFrame(table)
    row_labels = np.flip(np.arange(shape[0])) + 1
    df.index = row_labels

    print(tabulate(df, headers=headers))


def main():
    print("Calling main function in helpers.py")


if __name__ == "__main__":
    main()
