# SOURCE: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
import math
import numpy as np
import pandas as pd


def fwd_bkw_custom(observations, states, start_prob, trans_prob, emm_prob, end_st):
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
    for i, observation_i_plus in enumerate(reversed(observations[1:] + [None, ])):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)
        bkw.insert(0, b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)

    # Merging the two parts
    posterior = []
    for i in range(len(observations)):
        # this was added because the state transition distribution had rows with all 0s for the terminal state.
        # you have to set 1.0 to the terminal state when transitioning from the terminal state
        # if not p_fwd:
        #	posterior.append({st: 0.0 for st in states})
        # else:
        #	posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})
        # if the terminal state cycles to itself then you don't need to make the check above.
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})
    # print(p_fwd)
    # print(p_bkw)
    # https://davidamos.dev/the-right-way-to-compare-floats-in-python/
    assert math.isclose(p_fwd, p_bkw)
    return fwd, bkw, posterior


def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
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
    for i, observation_i_plus in enumerate(reversed(observations[1:] + (None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)
        bkw.insert(0, b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)

    # Merging the two parts
    posterior = []
    for i in range(len(observations)):
        # this was added because the state transition distribution had rows with all 0s for the terminal state.
        # you have to set 1.0 to the terminal state when transitioning from the terminal state
        # if not p_fwd:
        #	posterior.append({st: 0.0 for st in states})
        # else:
        #	posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})
        # if the terminal state cycles to itself then you don't need to make the check above.
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})
    print(p_fwd)
    print(p_bkw)
    # https://davidamos.dev/the-right-way-to-compare-floats-in-python/
    assert math.isclose(p_fwd, p_bkw)
    return fwd, bkw, posterior


def example():
    end_state = 'E'
    states = ('Healthy', 'Fever')
    observations = ('normal', 'cold', 'dizzy')
    # observations = ('dizzy', 'cold', 'normal')
    # observations = ('dizzy', 'dizzy', 'dizzy')

    start_probability = {'Healthy': 0.6, 'Fever': 0.4}
    transition_probability = {
        'Healthy': {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
        'Fever': {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
    }
    emission_probability = {
        'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
        'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
    }
    return fwd_bkw(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability,
                   end_state)


def example2():
    end_state = 'Z'
    states = ('A', 'B')
    observations = ('a', 'b', 'c', 'a')
    start_probability = {'A': 0.6, 'B': 0.4}
    transition_probability = {
        'A': {'A': 0.69, 'B': 0.30, 'Z': 0.01},
        'B': {'A': 0.40, 'B': 0.59, 'Z': 0.01},
    }
    emission_probability = {
        'A': {'a': 0.1, 'b': 0.4, 'c': 0.5},
        'B': {'a': 0.6, 'b': 0.3, 'c': 0.1},
    }
    return fwd_bkw(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability,
                   end_state)


def example3():
    end_state = '-1'
    states = ('0', '1', '2')
    observations = ('0', '1', '2', '0')
    start_probability = {'0': 0.6, '1': 0.4, '2': 0.0}
    transition_probability = {
        '0': {'0': 0.5, '1': 0.3, '2': 0.1, '-1': 0.1},
        '1': {'0': 0.4, '1': 0.4, '2': 0.1, '-1': 0.1},
        '2': {'0': 0.3, '1': 0.3, '2': 0.0, '-1': 0.4},
    }
    emission_probability = {
        '0': {'0': 0.1, '1': 0.4, '2': 0.5},
        '1': {'0': 0.6, '1': 0.3, '2': 0.1},
        '2': {'0': 0.3, '1': 0.3, '2': 0.3},
    }
    return fwd_bkw(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability,
                   end_state)


def example4():
    end_state = 7
    obs = [1, 1, 2, 0, 0]
    states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    start_p = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    trans_p = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.7, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ]
    trans_p[end_state][end_state] = 1.0  # setting terminal state
    emit_p = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ]
    return fwd_bkw(tuple(obs),
                   states,
                   start_p,
                   trans_p,
                   emit_p,
                   end_state)


def main():
    result = example()
    for line in result:
        print(*line)
    print('##FORWARD##')
    print(pd.DataFrame(result[0]))
    print('##BACKWARD##')
    print(pd.DataFrame(result[1]))
    print('##POSTERIOR##')
    print(pd.DataFrame(result[2]))


if __name__ == "__main__":
    main()
