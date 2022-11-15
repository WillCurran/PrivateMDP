import math
import heapq
import copy
import helpers as hlp

def path_cost(path, T):
    cost = 0
    print(path)
    for i in range(len(path)-1):
        cost += -1 * math.log(T[path[i]][path[i+1]])
    return cost

def path_prob(path, T):
    prob = 1
    for i in range(len(path)-1):
        prob *= T[path[i]][path[i+1]]

    return prob

def dijkstra(T, start, goal, prev_edge, max_repeats):
    distances = {}
    previous = {}
    for node in range(len(T)):
        distances[node] = math.inf
        previous[node] = -1

    min_heap = [(0, start)]
    distances[start] = 0
    previous[start] = -1
    max_time = 11

    num_pops = 0

    path = []
    while(len(min_heap) != 0):
        value, curr_state = heapq.heappop(min_heap)
        num_pops += 1
        # if(prev_edge != None and num_pops == 2):
        #     T[prev_edge[0]][prev_edge[1]] = prev_edge[2]
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
                    heapq.heappush(min_heap, (alt, node))
            node += 1

    curr_repeat = 1
    while(path == [] and curr_repeat != max_repeats):
        path = dijkstra_retry(T, start, goal, prev_edge, curr_repeat)
        if(len(path) <= curr_repeat):
            if(curr_repeat == max_repeats):
                path = []
        curr_repeat += 1

    return path

def dijkstra_retry(T, start, goal, prev_edge, curr_repeat):
    distances = {}
    previous = {}
    for node in range(len(T)):
        distances[node] = math.inf
        previous[node] = -1

    min_heap = [(0, start)]
    distances[start] = 0
    previous[start] = -1
    max_time = 11

    retry_count = 0
    num_pops = 0

    path = []

    T[prev_edge[0]][prev_edge[1]] = prev_edge[2]

    while(len(min_heap) != 0):
        value, curr_state = heapq.heappop(min_heap)
        num_pops += 1
        # if(prev_edge != None and num_pops == 2):
        #     T[prev_edge[0]][prev_edge[1]] = prev_edge[2]
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
                    heapq.heappush(min_heap, (alt, node))
            node += 1

    for i in range(0, curr_repeat):
        path = [start] + path

    if(len(path) == curr_repeat):
        return []

    return path

def kdijkstra_actions(T, start, goal, K, pi, max_repeats):
    dijkstra_res = dijkstra(T, start, goal, None, 1)
    A = [dijkstra_res]
    B = []

    #C = [([hlp.action_to_str(pi[node]) for node in dijkstra_res], path_prob(dijkstra_res, T))]

    C = [([int(pi[node]) for node in dijkstra_res], path_prob(dijkstra_res, T))]

    overall_prob = path_prob(dijkstra_res, T)

    prev_edge = None

    tCopy = copy.deepcopy(T)

    k = 1

    while(k < K):
        for i in range(len(A[k-1]) - 2):
            prev_edge = None
            spurNode = A[k-1][i]

            rootPath = A[k-1][0:i]

            for p in A:
                if rootPath == p[0:i]:
                    prev_edge = (p[i], p[i+1], T[p[i]][p[i+1]])
                    T[p[i]][p[i+1]] = 0
                    #T[p[i+1]][p[i]] = 0

            for rootNode in rootPath:
                if rootNode != spurNode:
                    # remove rootnode from trans_p
                    for i in range(len(T)):
                        T[i][rootNode] = 0
                        T[rootNode][i] = 0

            spurPath = dijkstra(T, spurNode, goal, prev_edge, max_repeats)

            if(len(spurPath) == 0):
                T = copy.deepcopy(tCopy)
                continue
            totalPath = rootPath + spurPath
            totalCost = path_cost(totalPath, tCopy)

            if B.count((totalCost, totalPath)) == 0:
                heapq.heappush(B, (totalCost, totalPath))
                #print("Total path " + str(totalPath))

            T = copy.deepcopy(tCopy)

        if len(B) == 0:
            print("break1")
            break

        # Check for a repeat
        # if its a repeat then add the costs together and only add one into A
        # increase K
        curr_prob = path_prob(B[0][1], T)
        overall_prob += curr_prob
        A.append(B[0][1])

        actions = [int(pi[node]) for node in B[0][1]]
        actions_str = [hlp.action_to_str(pi[node]) for node in B[0][1]]

        append = True

        for i in range(len(C)):
            curr_actions = C[i][0]
            if curr_actions == actions:
                C[i] = (actions, curr_prob + C[i][1])
                K += 1
                append = False
                break
        if append:
            C.append((actions, curr_prob))

        heapq.heappop(B)

        k += 1

        if(k == K and overall_prob <= .8):
            K *= 2

    T = tCopy

    # for i in range(len(A)):
    #     action_path = [pi[node] for node in A[i]]
    #     actions = actions + [[action_to_str(node) for node in action_path]]
    print(overall_prob)
    print(len(A))
    print(len(C))
    return C
