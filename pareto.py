import random

import matplotlib.pyplot as plt
import numpy as np


def plot_bounds(utility, lower_leakage, upper_leakage, title):
    """
    Function to plot the upper and lower bounds of expected leakage
    """
    # Plot the points
    plt.scatter(utility, lower_leakage, c='orange')
    plt.scatter(utility, upper_leakage, c='blue')

    # plot the lower bound as vertical lines
    plt.vlines(utility, lower_leakage, upper_leakage, color='black')

    # Set axis labels and title
    plt.xlabel('Utility')
    plt.ylabel('Expected Leakage')
    plt.title(title)

    # Show the plot
    plt.show()


def pareto_front(utility, lower_leakage, upper_leakage, title):
    """
    Function to calculate the Pareto front
    """
    sorted_indices_lower = sorted(
        range(len(lower_leakage)), key=lambda x: lower_leakage[x])
    sorted_indices_upper = sorted(
        range(len(upper_leakage)), key=lambda x: upper_leakage[x])
    pareto_front_lower = [sorted_indices_lower[0]]
    pareto_front_upper = [sorted_indices_upper[0]]
    for i in range(1, len(sorted_indices_lower)):
        if utility[sorted_indices_lower[i]] < utility[pareto_front_lower[-1]]:
            pareto_front_lower.append(sorted_indices_lower[i])

    for i in range(1, len(sorted_indices_upper)):
        if utility[sorted_indices_upper[i]] < utility[pareto_front_upper[-1]]:
            pareto_front_upper.append(sorted_indices_upper[i])

    pareto_front_utility = [utility[i] for i in pareto_front_lower]
    pareto_front_leakage_lower = [lower_leakage[i] for i in pareto_front_lower]
    pareto_front_leakage_upper = [upper_leakage[i] for i in pareto_front_upper]

    # Plot the Pareto front
    plt.scatter(utility, lower_leakage, c='orange')
    plt.scatter(utility, upper_leakage, c='blue')
    plt.scatter(pareto_front_utility, pareto_front_leakage_lower, c='red')
    plt.plot(pareto_front_utility, pareto_front_leakage_lower, '-r')

    pareto_front_utility = [utility[i] for i in pareto_front_upper]
    plt.scatter(pareto_front_utility, pareto_front_leakage_upper, c='green')
    plt.plot(pareto_front_utility, pareto_front_leakage_upper, '-g')

    # plot the lower bound as vertical lines
    plt.vlines(utility, lower_leakage, upper_leakage, color='black')

    plt.xlabel('Utility')
    plt.ylabel('Expected Leakage')
    plt.title(title)
    plt.show()


def pareto_front_separate(utility, lower_leakage, upper_leakage, title_1, title_2):
    """
    Function to calculate the Pareto front
    """
    sorted_indices_lower = sorted(
        range(len(lower_leakage)), key=lambda x: lower_leakage[x])
    sorted_indices_upper = sorted(
        range(len(upper_leakage)), key=lambda x: upper_leakage[x])
    pareto_front_lower = [sorted_indices_lower[0]]
    pareto_front_upper = [sorted_indices_upper[0]]
    for i in range(1, len(sorted_indices_lower)):
        if utility[sorted_indices_lower[i]] < utility[pareto_front_lower[-1]]:
            pareto_front_lower.append(sorted_indices_lower[i])

    for i in range(1, len(sorted_indices_upper)):
        if utility[sorted_indices_upper[i]] < utility[pareto_front_upper[-1]]:
            pareto_front_upper.append(sorted_indices_upper[i])

    pareto_front_utility = [utility[i] for i in pareto_front_lower]
    pareto_front_leakage_lower = [lower_leakage[i] for i in pareto_front_lower]
    pareto_front_leakage_upper = [upper_leakage[i] for i in pareto_front_upper]

    # Plot the Pareto front
    plt.figure(1)
    plt.scatter(utility, lower_leakage, c='orange')
    plt.scatter(pareto_front_utility, pareto_front_leakage_lower, c='red')
    plt.plot(pareto_front_utility, pareto_front_leakage_lower, '-r')
    # Set the y-axis grid lines to extend across the plot
    plt.grid(axis='y', linewidth=0.5)
    plt.xlabel('Utility')
    plt.ylabel('Expected Leakage')
    plt.title(title_1)

    plt.figure(2)
    plt.scatter(utility, upper_leakage, c='blue')
    pareto_front_utility = [utility[i] for i in pareto_front_upper]
    plt.scatter(pareto_front_utility, pareto_front_leakage_upper, c='green')
    plt.plot(pareto_front_utility, pareto_front_leakage_upper, '-g')
    plt.grid(axis='y', linewidth=0.5)
    plt.xlabel('Utility')
    plt.ylabel('Expected Leakage')

    print(pareto_front_leakage_upper)
    print(pareto_front_leakage_lower)

    plt.title(title_2)
    plt.show()
