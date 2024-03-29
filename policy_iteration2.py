#!/usr/bin/env python
# https://github.com/mpatacchiola/dissecting-reinforcement-learning
# MIT License
# Copyright (c) 2017 Massimiliano Patacchiola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Example of the policy iteration algorithm.
import helpers as hlp
import numpy as np


def return_policy_evaluation(p, u, r, T, gamma):
    for s in range(12):
        if not np.isnan(p[s]):
            v = np.zeros((1, 12))
            v[0, s] = 1.0
            action = int(p[s])
            u[s] = r[s] + gamma * \
                np.sum(np.multiply(u, np.dot(v, T[:,:, action])))
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
        # Expected utility of doing a in state s, according to T and u.
        actions_array[action] = np.sum(
            np.multiply(u, np.dot(v, T[:,:, action])))
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
            if p[counter] == -1:
                policy_string += " *  "
            elif p[counter] == 0:
                policy_string += " >  "
            elif p[counter] == 1:
                policy_string += " |  "
            elif np.isnan(p[counter]):
                policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)


def main_iterative(obs=[]):
    """Finding the solution using the iterative approach

    """
    gamma = 0.999
    iteration = 0
    print("================= LOADING TRANSITIONAL MATRIX ==================")
    T = np.load("T2.npy")
    print(T)
    print("====================== POLICY ITERATION ========================")
    # Generate the first policy randomly
    # Nan=Obstacle, -1=Terminal, 0=Forward, 1=Correct
    p = np.random.randint(0, 2, size=(12)).astype(np.float32)
    # Obstacles
    # p[5] = np.NaN
    # Terminal States
    p[7] = -1

    # Utility vectors
    u = np.array([0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0])

    # Reward vector
    r = np.array([-0.04, -0.04, -0.04, -0.04,
                  -0.04, -0.04, -0.04, +1.0,
                  -0.04, -0.04, -0.04, -0.04])

    while True:
        iteration += 1
        epsilon = 0.0001
        # 1- Policy evaluation
        u1 = u.copy()
        u = return_policy_evaluation(p, u, r, T, gamma)
        # Stopping criteria
        delta = np.absolute(u - u1).max()
        if delta < epsilon * (1 - gamma) / gamma:
            break
        for s in range(12):
            if not np.isnan(p[s]) and not p[s] == -1:
                v = np.zeros((1, 12))
                v[0, s] = 1.0
                # 2- Policy improvement
                a = return_expected_action(u, T, v)
                if a != p[s]:
                    p[s] = a
        print_policy(p, shape=(3, 4))

    print("================ POLICY ITERATION FINAL RESULT =================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("================================================================")
    hlp.print_world(u)
    print("================================================================")
    return(T, p, u, r, gamma)


def main():
    main_iterative()
    # main_linalg()


if __name__ == "__main__":
    main()
