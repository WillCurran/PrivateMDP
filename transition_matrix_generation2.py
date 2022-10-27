#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#This is a really raw script for generating the Transition matrix T
#for the cleaning robot world. It is a quick and dirty script but it works.
#I spent only 5 minutes on it and it can be improved to take into account
#the dimension of the world and the number of actions for generating
#different matrices. The matrix T has shape: state x state' x actions
#
#The script save the matrix in a numpy file "T.npy" which can be
#load using the function numpy.load("T.npy")

import numpy as np
import math

def return_transition(row, col, action, tot_row, tot_col):

    if(row > tot_row-1 or col > tot_col-1):
        print("ERROR: the index is out of range...")
        return None

    extended_world = np.zeros((tot_row+2, tot_col+2))

    #If the state is on the grey-obstacle it returns all zeros
    #if(row == 1 and col == 1): return extended_world[1:4, 1:5]
    #If the process is on the final reward state it returns zeros
    #if(row == 0 and col == 3): return extended_world[1:4, 1:5]
    #If the process is on the final punishment state then returns zeros
    if(row == 1 and col == 3): return extended_world[1:4, 1:5]

    if(action=="forward"):
        col += 1
        row += 1
        extended_world[row, col+1] = 0.4
        extended_world[row-1, col+1] = 0.3
        extended_world[row+1, col+1] = 0.3
    elif (action == "correct"):
        down = row < math.floor((tot_row-1)/2)
        col += 1
        row += 1
        if(down):
            extended_world[row + 1, col + 1] = 1
        else:
            extended_world[row - 1, col + 1] = 1
    #Reset the obstacle
    #if(extended_world[2, 2] != 0): extended_world[row, col] += extended_world[2, 2]
    #extended_world[2, 2] = 0.0

    #Control bouncing
    for row in range(0, 5):   
            if(extended_world[row, 0] != 0): extended_world[row, 1] += extended_world[row, 0]
            if(extended_world[row, 5] != 0): extended_world[row, 4] += extended_world[row, 5]
    for col in range(0, 6):
            if(extended_world[0, col] != 0): extended_world[1, col] += extended_world[0, col]
            if(extended_world[4, col] != 0): extended_world[3, col] += extended_world[4, col]

    return extended_world[1:4, 1:5]


def main():
    T = np.zeros((12, 12, 2))
    counter = 0
    for row in range(0, 3):
        for col in range(0, 4):
            line = return_transition(row, col, action="forward", tot_row=3, tot_col=4)
            T[counter, : , 0] = line.flatten()
            line = return_transition(row, col, action="correct", tot_row=3, tot_col=4)
            T[counter, : , 1] = line.flatten()

            counter += 1

    #print(T[:,:,3])
    #Get Transition dynamics at state 7
    u = np.array([[0.0, 0.0, 0.0 ,0.0,
                   0.0, 0.0, 0.0 ,1.0,
                   0.0, 0.0, 0.0 ,0.0]])

    #u = np.zeros((1, 12))

    print(np.dot(u, T[:,:,1]))

    print("Saving T in 'T2.npy' ...")
    np.save("T2", T)
    print("Done!")

if __name__ == "__main__":
    main()
