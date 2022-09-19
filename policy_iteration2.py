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

from cmath import isnan, nan
import numpy as np
import random
import math
import heapq 
import copy

CMP_DELTA = 0.000001





#adj[node] = [(edgeWeight, Neighbor)]
class Graph:
    def __init__(self):
        self.adj = {}
    
    def insert_node(self,name):
        self.adj[name] = []

    def remove_node(self,name):
        

        #needs to be changed later for efficiency
        #how can we quickly get the incoming edges?
        for node in self.adj:
            i = 0
            for weight,v in self.adj[node]:
                if v == name:
                    del self.adj[node][i]
                i += 1
        self.adj.pop(name)

    
    def insert_edge(self,n1,n2,weight):
        self.adj[n1] += [(weight, n2)]
    
    def remove_edge(self,n1,n2):
        i = 0
        for weight, v in self.adj[n1]:
            if v == n2:
                del self.adj[n1][i]
                return weight
            i+=1
        
    def print_adj(self):
        for node in self.adj:
            print(node + ": ", end='')
            print(self.adj[node])     
    
    def adjacent_edges(self,name):
        return self.adj[name]

    def getInt(name):
        if len(name) == 2:
            return int(name[1])
        return int(name[1:])

    def path_cost(self, path):
        cost = 0
        for i in range(len(path)-1):
            for weight, node in self.adj[path[i]]:
                if node == path[i+1]:
                    cost += weight
        return cost

def path_cost(path, T):
        cost = 0
        for i in range(len(path)-1):
            cost += -1 * math.log(T[path[i]][path[i+1]])

        return cost

def path_prob(path, T):
        prob = 1
        for i in range(len(path)-1):
            prob *= T[path[i]][path[i+1]]

        return prob

def trans_to_graph(trans_p):
    g = Graph()

    for i in range(len(trans_p)):
        for j in range(len(trans_p[i])):
            if i == 0:
                g.insert_node("v"+str(j))
            if trans_p[i][j] != 0:
                g.insert_edge("v"+str(i),"v"+str(j), - math.log(trans_p[i][j]))
    g.print_adj()
    return g

            


def action_to_str(a):
    if a == -1:
        return "DONE"
    elif a == 0:
        return ">"
    elif a == 1:
        return "|"
    return "#"

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
            if(p[counter] == -1): policy_string += " *  "            
            elif(p[counter] == 0): policy_string += " >  "
            elif(p[counter] == 1): policy_string += " |  "
            elif(np.isnan(p[counter])): policy_string += " #  "
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

def dijkstra_actions(T, start, goal,p):
    path = dijkstra(T, start, goal)
    path = [p[node] for node in path]
    actions = [action_to_str(node) for node in path]
    return actions

def dijkstra(T, start, goal):
    distances = {}
    previous = {}
    for node in range(len(T)):
        distances[node] = math.inf
        previous[node] = -1

    min_heap = [(0,start)]
    distances[start] = 0
    previous[start] = -1
    max_time = 11

    path = []
    while(len(min_heap) != 0):
        value, curr_state = heapq.heappop(min_heap)
        if(curr_state == goal):
            while curr_state != -1:
                #print("State: " + str(curr_state))
                path = [curr_state] + path
                curr_state = previous[curr_state]
            break
        curr_adj = T[curr_state]
        node = 0
        for weight in curr_adj:
            if weight != 0:
                weight = -1 * math.log(weight)
                alt = distances[curr_state] + weight 
                if alt < distances[node]:
                    distances[node] = alt
                    previous[node] = curr_state
                    heapq.heappush(min_heap,(alt, node))
            node += 1
    

    return path
#SOURCE: https://en.wikipedia.org/wiki/Yen%27s_algorithm
def kdijkstra(T,start,goal,K):
    A = [(dijkstra(T,start,goal))]

    B = []

    tCopy = copy.deepcopy(T)
    for k in range(1,K):
        for i in range(len(A[k-1]) - 2):

            spurNode = A[k-1][i]

            rootPath = A[k-1][0:i]
   
            for p in A:
                if rootPath == p[0:i]:
                    T[p[i]][p[i+1]] = 0
                    T[p[i+1]][p[i]] = 0
                    

            for rootNode in rootPath:
                if rootNode != spurNode:
                    #remove rootnode from trans_p
                    for i in range(len(T)):
                        T[i][rootNode] = 0
                        T[rootNode][i] = 0
            
            spurPath = dijkstra(T,spurNode,goal)
            
            if(len(spurPath) == 0):
                T = copy.deepcopy(tCopy)
                continue
            totalPath = rootPath + spurPath
            totalCost = path_cost(totalPath, tCopy)

            if B.count((totalCost, totalPath)) == 0:
                heapq.heappush(B, (totalCost, totalPath))
            
            T = copy.deepcopy(tCopy)
            

        if len(B) == 0:
            break

        A += [B[0][1]]
        heapq.heappop(B)


    T = tCopy
    return A



def kdijkstra_actions(T,start,goal,K,pi):
    A = [(dijkstra(T,start,goal))]

    B = []

    tCopy = copy.deepcopy(T)
    for k in range(1,K):
        for i in range(len(A[k-1]) - 2):

            spurNode = A[k-1][i]

            rootPath = A[k-1][0:i]
   
            for p in A:
                if rootPath == p[0:i]:
                    T[p[i]][p[i+1]] = 0
                    T[p[i+1]][p[i]] = 0
                    

            for rootNode in rootPath:
                if rootNode != spurNode:
                    #remove rootnode from trans_p
                    for i in range(len(T)):
                        T[i][rootNode] = 0
                        T[rootNode][i] = 0
            
            spurPath = dijkstra(T,spurNode,goal)
            
            if(len(spurPath) == 0):
                T = copy.deepcopy(tCopy)
                continue
            totalPath = rootPath + spurPath
            totalCost = path_cost(totalPath, tCopy)

            if B.count((totalCost, totalPath)) == 0:
                heapq.heappush(B, (totalCost, totalPath))
            
            T = copy.deepcopy(tCopy)
            

        if len(B) == 0:
            break

        A += [B[0][1]]
        heapq.heappop(B)


    T = tCopy
    action_path = []
    actions = []
    for i in range(len(A)):
        action_path = [pi[node] for node in A[i]]
        actions = actions + [action_to_str(node) for node in action_path]
    return A


def main_iterative():
    """Finding the solution using the iterative approach

    """
    gamma = 0.999
    iteration = 0
    T = np.load("T2.npy")

    #Generate the first policy randomly
    # Nan=Obstacle, -1=Terminal, 0=Forward, 1=Correct
    p = np.random.randint(0, 2, size=(12)).astype(np.float32)
    #p[5] = np.NaN
    #p[3] = p[7] = -1
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

    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3,4))
    print("===================================================")
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
    

    
    
    # print("=======================Graph==========================")
   
    # graph = trans_to_graph(trans_p)
    print("=======================Dijkstra==========================")
    #print(trans_p)

    # g = trans_to_graph(trans_p)
    # D = dijkstra(g,"v4","v7")
    D = (dijkstra(trans_p,4,7))
    print(D)
    print(path_prob(D,trans_p))

    print("=======================KDijkstra==========================")

    A = kdijkstra_actions(trans_p, 4, 7, 10,p)

    print(A)


    print("====================== A Priori Analysis ====================")
    interesting_time = 4
    interesting_state = 3
    prior_expected_visits = get_expected_visits(states, start_p, T, p, interesting_time)
    print("Expected visits: \n" + ', '.join(["%.2f" % prior_expected_visits[st] for st in states]))
    print("Sum of expected visits should = 1 + t. %.2f == %d." % (sum(prior_expected_visits), 1+interesting_time) )
    print("=================== EXEC  POLICY ==================")
    obs = execute_policy(p, T, start_pos, 12)
    s = "["
    for a in obs:
        s += action_to_str(a) + ", "
    if len(s) > 1:
        print(s[:-2] + "]")
    else:
        print("[]")
    # obs needs positive indices for viterbi alg implementation below
    obs = [obs[i]+1 for i in range(len(obs))]
    print("====================== VITERBI ====================")
    (dp_table, max_path_prob) = viterbi(obs, states, start_p, trans_p, emit_p)
    if interesting_time >= len(dp_table):
        print("Actual execution did not go as long as %d steps. How to handle information gain here?" % interesting_time)
    else:
        post_expected_visits = [dp_table[interesting_time][st]["prob"] for st in states]
        print("Actual expected visits given single execution: \n" + ', '.join(["%.2f" % post_expected_visits[st] for st in states]))
        print("====================== INFORMATION GAIN ====================")
        ig = information_gain(prior_expected_visits, post_expected_visits, interesting_state, max_path_prob)
        print("Information Gain on state=%d and time=%d: %.2f" % (interesting_state, interesting_time, ig))

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

def main():
    main_iterative()
    #main_linalg()


if __name__ == "__main__":
    main()