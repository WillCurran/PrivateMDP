# https://github.com/aehuynh/hidden-markov-model

import numpy as np
import pandas as pd

class HMM:
    """
    Order 1 Hidden Markov Model

    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector

    Common Variables
    ----------------
    obs_seq : list of int
        list of observations (represented as ints corresponding to output
        indexes in B) in order of appearance
    T : int
        number of observations in an observation sequence
    N : int
        number of states
    """

    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def _forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((N, T))
        F[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.dot(F[:, t - 1], (self.A[:, n])) * self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N, T))
        X[:, -1:] = 1

        for t in reversed(range(T - 1)):
            for n in range(N):
                X[n, t] = np.sum(X[:, t + 1] * self.A[n, :] * self.B[:, obs_seq[t + 1]])

        return X

    def forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((T, N))

        F[0, :] = self.pi * self.B[:, obs_seq[0]]
        F[0] = F[0] / F[0].sum()

        for t in range(1, T):
            for n in range(N):
                F[t, n] = np.dot(F[t - 1, :], (self.A[:, n])) * self.B[n, obs_seq[t]]
            # Normalize
            F[t] = F[t] / F[t].sum()

        first = np.array([self.pi])

        return np.concatenate([first, F])

    def backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((T, N))
        X[-1:, :] = 1

        for t in reversed(range(T - 1)):
            for n in range(N):
                X[t, n] = np.sum(X[t + 1, :] * self.A[n, :] * self.B[:, obs_seq[t + 1]])
            # Normalize
            X[t] = X[t] / X[t].sum()
        first = np.zeros((1, N))

        for n in range(N):
            first[0][n] = np.sum(X[0, :] * self.A[n, :] * self.B[:, obs_seq[0]])
        first = first / first.sum()
        return np.concatenate([first, X])

    def forward_backward(self, obs_seq):
        # Compute the forward probabilities
        forward = self.forward(obs_seq)
        # Compute the backward probabilities
        backward = self.backward(obs_seq)
        # Compute the posterior marginals
        posteriors = forward * backward / np.sum(forward * backward, axis=1, keepdims=True)
        return posteriors

    def observation_prob(self, obs_seq):
        """ P( entire observation sequence | A, B, pi ) """
        return np.sum(self._forward(obs_seq)[:, -1])

    def state_path(self, obs_seq):
        """
        Returns
        -------
        V[last_state, -1] : float
            Probability of the optimal state path
        path : list(int)
            Optimal state path for the observation sequence
        """
        V, prev = self.viterbi(obs_seq)

        # Build state path with greatest probability
        last_state = np.argmax(V[:, -1])
        path = list(self.build_viterbi_path(prev, last_state))

        return V[last_state, -1], reversed(path)

    def viterbi(self, obs_seq):
        """
        Returns
        -------
        V : numpy.ndarray
            V [s][t] = Maximum probability of an observation sequence ending
                       at time 't' with final state 's'
        prev : numpy.ndarray
            Contains a pointer to the previous state at t-1 that maximizes
            V[state][t]
        """
        N = self.A.shape[0]
        T = len(obs_seq)
        prev = np.zeros((T - 1, N), dtype=int)

        # DP matrix containing max likelihood of state at a given time
        V = np.zeros((N, T))
        V[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                seq_probs = V[:, t - 1] * self.A[:, n] * self.B[n, obs_seq[t]]
                prev[t - 1, n] = np.argmax(seq_probs)
                V[n, t] = np.max(seq_probs)

        return V, prev

    def build_viterbi_path(self, prev, last_state):
        """Returns a state path ending in last_state in reverse order."""
        T = len(prev)
        yield (last_state)
        for i in range(T - 1, -1, -1):
            yield (prev[i, last_state])
            last_state = prev[i, last_state]

    def baum_welch_train(self, obs_seq):

        N = self.A.shape[0]
        T = len(obs_seq)

        forw = self._forward(obs_seq)
        back = self._backward(obs_seq)

        # P( entire observation sequence | A, B, pi )
        obs_prob = np.sum(forw[:, -1])
        if obs_prob <= 0:
            raise ValueError("P(O | lambda) = 0. Cannot optimize!")

        xi = np.zeros((T - 1, N, N))
        for t in range(xi.shape[0]):
            xi[t, :, :] = self.A * forw[:, [t]] * self.B[:, obs_seq[t + 1]] * back[:, t + 1] / obs_prob

        gamma = forw * back / obs_prob

        # Gamma sum excluding last column
        gamma_sum_A = np.sum(gamma[:, :-1], axis=1, keepdims=True)
        # Vector of binary values indicating whether a row in gamma_sum is 0.
        # If a gamma_sum row is 0, save old rows on update
        rows_to_keep_A = (gamma_sum_A == 0)
        # Convert all 0s to 1s to avoid division by zero
        gamma_sum_A[gamma_sum_A == 0] = 1.
        next_A = np.sum(xi, axis=0) / gamma_sum_A

        gamma_sum_B = np.sum(gamma, axis=1, keepdims=True)
        rows_to_keep_B = (gamma_sum_B == 0)
        gamma_sum_B[gamma_sum_B == 0] = 1.

        obs_mat = np.zeros((T, self.B.shape[1]))
        obs_mat[range(T), obs_seq] = 1
        next_B = np.dot(gamma, obs_mat) / gamma_sum_B

        # Update model
        self.A = self.A * rows_to_keep_A + next_A
        self.B = self.B * rows_to_keep_B + next_B
        self.pi = gamma[:, 0] / np.sum(gamma[:, 0])


def example():
    umbrella_transition = [[0.7, 0.3], [0.3, 0.7]]
    umbrella_sensor = [[0.9, 0.1], [0.2, 0.8]]
    umbrella_initial = [0.5, 0.5]
    umbrellaHMM = HMM(np.array(umbrella_transition), np.array(umbrella_sensor), np.array(umbrella_initial))
    # {umbrella, umbrella, no umbrella, umbrella, umbrella}
    umbrella_evidence = [0, 0, 1, 0, 0]
    # result = umbrellaHMM.forward(umbrella_evidence)
    # result = umbrellaHMM.backward(umbrella_evidence)
    result = umbrellaHMM.forward_backward(umbrella_evidence)
    return (result)


def main():
    result = example()
    print(result)

    # Create a 2D numpy array with random values
    # array = np.random.rand(5, 3)
    # Print the original array
    # print("Original array:")
    # print(array)
    # Normalize the array by dividing each column by the sum of all the elements in that column
    # array = array / array.sum(axis=0)
    # Print the normalized array
    # print("Normalized array:")
    # print(array)
    # Check if the sum of the elements in each column of the normalized array is equal to 1
    # print("Sum of the elements in each column of the normalized array:", array.sum(axis=0))


if __name__ == "__main__":
    main()
