import math
import os

import jpype

import helpers as h
import pandas as pd
import policy_iteration as russel_norvig_world
import dijkstras as dk

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


def trans_to_graph(trans_p, num):
    g = []
    for i in range(len(trans_p)):
        for j in range(len(trans_p[i])):
            if trans_p[i][j] != 0:
                g.append((str(i), str(j), str(-math.log(trans_p[i][j]))))

    df = pd.DataFrame(g)
    df.to_csv("eppstein_files/eppstein_graph" + str(num%1) +
              ".csv", index=False, sep=" ", header=False)
    

    return g


def run_eppstein():
    jpype.startJVM()

    T, p, u, r, gamma, p_hist = russel_norvig_world.main_iterative()

    start_state = 8
    states = [i for i in range(12)]
    actions = [i for i in range(4)]
    start_p = [0.0 for i in range(12)]
    start_p[start_state] = 1.0

    policies = h.enumerate_policies(states, actions, [5], [3, 7])
    policies_outputs = []
    length = len(policies)
    for i in range(length):
        p = policies[i]
        print(i)
        #russel_norvig_world.print_policy(p,(3,4))
        states, start_p, trans_p, emit_p = h.to_hidden_markov_model(
            T, p, 12, 4, start_state)

        #Add viterbi to check if possible
        #if not continue
        d = dk.dijkstra2(trans_p,start_state,3)
        if(len(d) == 0):
            print("FAILED")
            continue
        classpath = os.getcwd() + "/k-shortest-paths-master/out/production/k-shortest-paths-master"
        jpype.addClassPath(classpath)

        classname = "edu.ufl.cise.bsmock.graph.ksp.test.TestEppstein"
        MyClass = jpype.JClass(classname)
        output_stream = jpype.JPackage('java.io').ByteArrayOutputStream()
        jpype.JPackage('java.lang').System.setOut(jpype.JPackage('java.io').PrintStream(output_stream))

        trans_to_graph(trans_p, i)
        start = str(start_state)
        end = str(3)
        k = str(1000)
        args = ["eppstein_files/eppstein_graph" +
                str(i%1) + ".csv", start, end, k]
        # Call a method in the Java class with arguments
        #print('Run Eppstein Java Code')
        MyClass.main(args)
        result = str(output_stream.toString())
        #print('result')
        data = []
        
        #Parse result string to create array of tuples
        #(probability, states, actions)
        for line in result.splitlines()[:-1]:
            parts = line.split(':')
            num = float(parts[0])
            arr_str = parts[1].strip()[1:-1]
            arr = [int(x) for x in arr_str.split(',')]
            actions = [p[x] for x in arr]
            data.append((num, arr, actions))
        
        # for t in data:
            
        #     print(t)

        policies_outputs.append(data)
        #print('================================================================')
        

    # Stop the JVM
    jpype.shutdownJVM()

    return policies_outputs


def main():

    run_eppstein()


if __name__ == "__main__":
    main()
