import policy_iteration as russel_norvig_world
import helpers as h
def extract_data(filename,pi):
    data = []
    obs = []
    with open(filename, 'r') as f:
        for line in f:
            line_split = line.split(': ')
            float_num = float(line_split[0])
            int_list = []
            for x in line_split[1][1:-2].split(', '):
                if x:
                    int_list.append(int(x))
            actions = [int(pi[node]) for node in int_list]
            data.append((int_list, float_num))
            append = True

            for i in range(len(obs)):
                curr_actions = obs[i][0]
                if curr_actions == actions:
                    obs[i] = (actions, float_num + obs[i][1])
                    append = False
                    break
            if append:
                obs.append((actions,float_num))
            
    return data, obs


T, p, u, r, gamma = russel_norvig_world.main_iterative()

filename = 'russelworld.txt'
result, obs = extract_data(filename,p)
print(*obs, sep="\n")
