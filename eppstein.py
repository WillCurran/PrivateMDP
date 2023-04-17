from io import StringIO
import math
import math
import os
import time

import jpype

import dijkstras as dk
import helpers as h
import numpy as np
import pandas as pd
import policy_iteration as russel_norvig_world


graph_path = "eppstein_graphs/graph.csv"


# def eppstein_output_to_list(p, output):
#     result = []
#     # Parse result string to create array of tuples
#     # (probability, states, actions)
#     for line in output.splitlines()[:-1]:
#         parts = line.split(':')
#         num = float(parts[0])
#         arr_str = parts[1].strip()[1:-1]
#         arr = [int(x) for x in arr_str.split(',')]
#         actions = [int(p[x]) for x in arr]
#         result.append((num, arr, actions))
#     return result


def eppstein_output_to_list(p, output):
    # Parse result string to create list of tuples (probability, states, actions)
    result = [(float(line.split(':')[0]),
               [int(x) for x in line.split(':')[1].strip()[1:-1].split(',')],
               [int(p[int(x)])
                for x in line.split(':')[1].strip()[1:-1].split(',')]
               ) for line in output.splitlines()[:-1]]
    return result


def extract_data(filename, pi):
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
                obs.append((actions, float_num))

    return data, obs


# def trans_to_graph(trans_p, path):
#     g = []
#     for i in range(len(trans_p)):
#         for j in range(len(trans_p[i])):
#             if trans_p[i][j] != 0:
#                 g.append((str(i), str(j), str(-math.log(trans_p[i][j]))))
#
#     df = pd.DataFrame(g)
#     df.to_csv(path, index=False, sep=" ", header=False)
#
#     return g


def trans_to_graph(trans_p, path):
    row, col = np.where(trans_p != 0)
    data = -np.log(trans_p[row, col])
    g = np.column_stack((row.astype(str), col.astype(str), data.astype(str)))

    df = pd.DataFrame(g)
    df.to_csv(path, index=False, sep=" ", header=False)

    return g


def eppstein(trans_p, p, start_state, end_state, k):
    result = []

    # if not continue
    d = dk.dijkstra2(trans_p, start_state, end_state)
    if(len(d) == 0):
        print("FAILED")
        russel_norvig_world.print_policy(p, (3, 4))
        return result
    classpath = os.getcwd() + "/k-shortest-paths-master/out/production/k-shortest-paths-master"
    jpype.addClassPath(classpath)

    classname = "edu.ufl.cise.bsmock.graph.ksp.test.TestEppstein"
    MyClass = jpype.JClass(classname)
    output_stream = jpype.JPackage('java.io').ByteArrayOutputStream()
    jpype.JPackage('java.lang').System.setOut(
        jpype.JPackage('java.io').PrintStream(output_stream))

    trans_to_graph(trans_p, graph_path)
    args = [graph_path, str(start_state), str(end_state), str(k)]
    MyClass.main(args)
    output_string = str(output_stream.toString())
    result = eppstein_output_to_list(p, output_string)
    print(output_string)
    print('end')
    return result


def main():
    jpype.startJVM()
    print('running eppstein on optimal policy')
    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()
    start_state = 8
    end_state = 3
    k = 10000
    states, start_p, trans_p, emit_p = h.to_hidden_markov_model(
        T, p, 12, 4, start_state)
    print('result')
    start_time = time.time()
    end_time = time.time()
    execution_time = end_time - start_time
    result = eppstein(trans_p, p, start_state, end_state, k)
    print(result[0])
    print(f"Execution time: {execution_time} seconds")
    jpype.shutdownJVM()


if __name__ == "__main__":
    main()
