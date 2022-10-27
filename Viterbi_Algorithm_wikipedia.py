#SOURCE: https://en.wikipedia.org/wiki/Viterbi_algorithm
import numpy as np
import pandas as pd
def viterbi_custom(obs, states, start_p, trans_p, emit_p):
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

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0] [st] = {"prob": start_p[st] * emit_p[st] [obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1] [states[0]] ["prob"] * trans_p[states[0]] [st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1] [prev_st] ["prob"] * trans_p[prev_st] [st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st] [obs[t]]
            V[t] [st] = {"prob": max_prob, "prev": prev_st_selected}

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

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1] [previous] ["prev"])
        previous = V[t + 1] [previous] ["prev"]

    print ("The steps of states are " + "".join(str(opt)) + " with highest probability of %s" % max_prob)
    return V

def dptable(V):
    # Print a table of steps from dictionary
    yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%lf" % v[state] ["prob"]) for v in V)

def example():
	obs = ("normal", "cold", "dizzy")
	states = ("Healthy", "Fever")
	start_p = {"Healthy": 0.6, "Fever": 0.4}
	trans_p = {
		"Healthy": {"Healthy": 0.7, "Fever": 0.3},
		"Fever": {"Healthy": 0.4, "Fever": 0.6},
	}
	emit_p = {
		"Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
		"Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
	}
	return viterbi(obs, states, start_p, trans_p, emit_p)

def example2():
    states = ('A', 'B')
    observations = ('a', 'b', 'c', 'a')
    start_probability = {'A': 0.6, 'B': 0.4}
    transition_probability = {
       'A' : {'A': 0.7, 'B': 0.3},
       'B' : {'A': 0.4, 'B': 0.6},
       }
    emission_probability = {
       'A' : {'a': 0.1, 'b': 0.4, 'c': 0.5},
       'B' : {'a': 0.6, 'b': 0.3, 'c': 0.1},
       }
    return viterbi(observations, states, start_probability, transition_probability, emission_probability)

def example3():
    states = ('0', '1','2')
    observations = ('0', '1', '2', '1')
    start_probability = {'0': 0.6, '1': 0.4, '2':0.0}
    transition_probability = {
       '0' : {'0': 0.5, '1': 0.3, '2':0.1},
       '1' : {'0': 0.4, '1': 0.4, '2':0.1},
       '2' : {'0': 0.3, '1': 0.3, '2':0.0},
       }
    emission_probability = {
       '0' : {'0': 0.1, '1': 0.4, '2': 0.5},
       '1' : {'0': 0.6, '1': 0.3, '2': 0.1},
       '2' : {'0': 0.3, '1': 0.3, '2': 0.3},
       }
    return viterbi(observations, states, start_probability, transition_probability, emission_probability)

def example4():
    end_state = 7
    obs = [1,1,2,0,0]
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
    trans_p[end_state][end_state]= 1.0 #setting terminal state?
    emit_p = [
		[0.0, 0.0, 1.0, 0.0, 0.0], 
		[0.0, 1.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 1.0, 0.0, 0.0], 
		[0.0, 0.0, 1.0, 0.0, 0.0], 
		[0.0, 1.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 1.0, 0.0, 0.0], 
		[0.0, 1.0, 0.0, 0.0, 0.0], 
		[1.0, 0.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 1.0, 0.0, 0.0], 
		[0.0, 1.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 1.0, 0.0, 0.0], 
		[0.0, 0.0, 1.0, 0.0, 0.0]
    ]
    return viterbi(obs, states, start_p, trans_p, emit_p)

def main():
	result = example4()
	print(pd.DataFrame(result))

if __name__ == "__main__":
    main()