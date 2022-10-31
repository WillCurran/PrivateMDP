#Original Source code from wikipedia
#states = ('Healthy', 'Fever')
#end_state = 'E'
 
#observations = ('normal', 'cold', 'dizzy')
 
#start_probability = {'Healthy': 0.6, 'Fever': 0.4}
 
#transition_probability = {
#   'Healthy' : {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
#   'Fever' : {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
#   }
 
#emission_probability = {
#   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
#   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
#   }
   
states = ('Rainy', 'Sunny')
end_state = 'E'

observations = ('walk', 'shop', 'clean')
 
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
 
transition_probability = {
   'Rainy' : {'Rainy': 0.69, 'Sunny': 0.3, 'E': 0.01},
   'Sunny' : {'Rainy': 0.4, 'Sunny': 0.59, 'E': 0.01},
   }
 
emission_probability = {
   'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
   'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
   }
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
    for i, observation_i_plus in enumerate(reversed(observations[1:] + (None,))):
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
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    assert p_fwd == p_bkw
    return fwd, bkw, posterior
    
def example():
    return fwd_bkw(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability,
                   end_state)
result = example()
for line in result:
	print(*line)
print('####')
print(result[0])
print('####')
print(result[1])
print('####')
print(result[2])