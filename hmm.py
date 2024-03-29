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
                F[n, t] = np.dot(F[:, t - 1], (self.A[:, n])
                                 ) * self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N, T))
        X[:, -1:] = 1

        for t in reversed(range(T - 1)):
            for n in range(N):
                X[n, t] = np.sum(X[:, t + 1] * self.A[n,:]
                                 * self.B[:, obs_seq[t + 1]])

        return X
    
    def _forward_backward(self, obs_seq):

        alpha = self._forward(obs_seq)
        beta = self._backward(obs_seq)
        
        gamma = alpha * beta
        gamma /= gamma.sum(axis=0)
        
        return gamma

    """ 
        Modified 
        Here, we are checking if the denominator sum_F is equal to zero before dividing. 
        If it is zero, we set the corresponding row in F to be a uniform distribution.
        Includes initial state probabilities and is normalized at each step.
    """

    def forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((T, N))

        F[0,:] = self.pi * self.B[:, obs_seq[0]]
        F[0] = F[0] / F[0].sum()

        for t in range(1, T):
            for n in range(N):
                F[t, n] = np.dot(F[t - 1,:], self.A[:, n]) * self.B[n, obs_seq[t]]
            # Normalize
            sum_F = F[t].sum()
            if sum_F == 0:
                F[t] = np.ones((N,)) / N
            else:
                F[t] = F[t] / sum_F

        first = np.array([self.pi])
        return np.concatenate([first, F])

    """
        Modified  
        Here, we are checking if the denominator sum_X is equal to zero before dividing. 
        If it is zero, we set the corresponding row in X to be a uniform distribution.
        Includes initial state probabilities and is normalized at each step.
    """

    def backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((T, N))
        X[-1:,:] = 1

        for t in reversed(range(T - 1)):
            for n in range(N):
                X[t, n] = np.sum(X[t + 1,:] * self.A[n,:] * self.B[:, obs_seq[t + 1]])
            # Normalize
            sum_X = X[t].sum()
            if sum_X == 0:
                X[t] = np.ones((N,)) / N
            else:
                X[t] = X[t] / sum_X
        first = np.zeros((1, N))

        for n in range(N):
            first[0][n] = np.sum(X[0,:] * self.A[n,:]
                                 * self.B[:, obs_seq[0]])

        first = first / first.sum()
        return np.concatenate([first, X])

    """
        This code first computes the product of the forward and backward probabilities, 
        and then computes their sum along the time axis.
        If the sum is nonzero, the posteriors are computed as 
        the product of the forward and backward probabilities, normalized by their sum. 
        If the sum is zero, the posteriors are set to be uniform.
    """

    def forward_backward(self, obs_seq):
        # Compute the forward probabilities
        forward = self.forward(obs_seq)
        # Compute the backward probabilities
        backward = self.backward(obs_seq)
        # Compute the posterior marginals
        posteriors_sum = np.sum(forward * backward, axis=1, keepdims=True)
        # Replace zeros with ones to avoid division by zero
        posteriors_sum[posteriors_sum == 0] = 1
        # Use a uniform distribution as a substitute for zero
        uniform_dist = np.full_like(forward, 1 / forward.shape[1])
        posteriors = np.where(posteriors_sum == 0, uniform_dist,
                              forward * backward / posteriors_sum)
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
            xi[t,:,:] = self.A * forw[:, [t]] * \
                self.B[:, obs_seq[t + 1]] * back[:, t + 1] / obs_prob

        gamma = forw * back / obs_prob

        # Gamma sum excluding last column
        gamma_sum_A = np.sum(gamma[:,:-1], axis=1, keepdims=True)
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

    def viterbi_beam(self, obs, n):
        N = self.A.shape[0]
        T = len(obs)
        # Initialize the first column of the trellis.
        trellis = [{state: {'prob': self.pi[state] * self.B[state]
                            [obs[0]], 'prev': None} for state in range(N)}]
        if n == 1:
            V, prev = self.viterbi(obs)
            last_state = np.argmax(V[:, -1])
            path = list(self.build_viterbi_path(prev, last_state))[::-1]
            return [path]
        # Iterate over the observations, computing the most likely state at each time step.
        for i in range(1, T):
            # Use beam search to limit the number of states considered at each time step.
            beam_width = n if n < N else N
            candidates = sorted(
                trellis[-1].items(), key=lambda x: x[1]['prob'], reverse=True)[:beam_width]

            # Update the trellis with the most likely states for this observation.
            trellis.append({})
            for state, prev_state_info in candidates:
                max_prob = float('-inf')
                max_prev = None
                for prev_state, prev_state_info in trellis[-2].items():
                    prob = prev_state_info['prob'] * \
                        self.A[prev_state][state] * self.B[state][obs[i]]
                    if prob > max_prob:
                        max_prob = prob
                        max_prev = prev_state
                trellis[-1][state] = {'prob': max_prob, 'prev': max_prev}

        # Find the n most likely state sequences by backtracking through the trellis.
        sequences = []
        beam_width = n if n < N else N
        candidates = sorted(
            trellis[-1].items(), key=lambda x: x[1]['prob'], reverse=True)[:beam_width]
        for state, state_info in candidates:
            sequence = [state]
            prev_state = trellis[-1][state]['prev']
            for i in range(T - 2, -1, -1):
                sequence.insert(0, prev_state)
                prev_state = trellis[i][prev_state]['prev']
            sequences.append(sequence)
        return sequences


def example():
    umbrella_transition = [[0.7, 0.3, 0], [0.3, 0.7, 0], [0, 0, 0]]
    umbrella_sensor = [[0.9, 0.1, 0], [0.2, 0.8, 0], [0, 0, 0]]
    umbrella_initial = [0.5, 0.5, 0]
    umbrellaHMM = HMM(np.array(umbrella_transition), np.array(
        umbrella_sensor), np.array(umbrella_initial))
    # {umbrella, umbrella, no umbrella, umbrella, umbrella}
    umbrella_evidence = [0, 0, 1, 0, 0]
    result_1 = umbrellaHMM.forward(umbrella_evidence)
    result_2 = umbrellaHMM.backward(umbrella_evidence)
    result_3 = umbrellaHMM._forward(umbrella_evidence).T
    result_4 = umbrellaHMM._backward(umbrella_evidence).T
    result_5 = umbrellaHMM.forward_backward(umbrella_evidence)
    result_6 = umbrellaHMM._forward_backward(umbrella_evidence).T
    V, prev = umbrellaHMM.viterbi(umbrella_evidence)
    last_state = np.argmax(V[:, -1])
    path = list(umbrellaHMM.build_viterbi_path(prev, last_state))[::-1]
    print(path)
    result_7 = umbrellaHMM.viterbi_beam(umbrella_evidence, 10)
    return (result_1, result_2, result_3, result_4, result_5, result_6)


def main():
    result = example()
    labels = ['Modified Forward Algorithm', 'Modified Backward Algorihm', 'Original Forward Algorithm', 'Original Backward Algorithm', 'Forward Backward Algorithm', 'Orginal Forward Backward Algorithm', 'Viterbi Beam']

    for i in range(len(result)):
         print(labels[i])
         print(result[i])

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
