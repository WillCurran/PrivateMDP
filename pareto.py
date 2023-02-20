import matplotlib.pyplot as plt
import random
import numpy as np


def pareto_front(utility,lower_leakage,upper_leakage,title):
    """
    Function to calculate the Pareto front
    """
    sorted_indices_lower = sorted(range(len(lower_leakage)), key=lambda x: lower_leakage[x])
    sorted_indices_upper = sorted(range(len(upper_leakage)), key=lambda x: upper_leakage[x])
    pareto_front_lower = [sorted_indices_lower[0]]
    pareto_front_upper = [sorted_indices_upper[0]]
    for i in range(1, len(sorted_indices_lower)):
        if utility[sorted_indices_lower[i]] >= utility[pareto_front_lower[-1]]:
            pareto_front_lower.append(sorted_indices_lower[i])


    for i in range(1, len(sorted_indices_upper)):
        if utility[sorted_indices_upper[i]] >= utility[pareto_front_upper[-1]]:
            pareto_front_upper.append(sorted_indices_upper[i])


    pareto_front_utility = [utility[i] for i in pareto_front_lower]
    pareto_front_leakage_lower = [lower_leakage[i] for i in pareto_front_lower]
    pareto_front_leakage_upper = [upper_leakage[i] for i in pareto_front_upper]

    # Plot the Pareto front
    plt.scatter(utility, lower_leakage)
    plt.scatter(utility, upper_leakage, c='blue')
    plt.scatter(pareto_front_utility, pareto_front_leakage_lower, c='red')
    plt.plot(pareto_front_utility, pareto_front_leakage_lower, '-r')

    plt.scatter(pareto_front_utility, pareto_front_leakage_upper, c='green')
    plt.plot(pareto_front_utility, pareto_front_leakage_upper, '-g')

    plt.xlabel('Utility')
    plt.ylabel('Expected Leakage')
    plt.title(title)
    plt.show()
    return pareto_front



