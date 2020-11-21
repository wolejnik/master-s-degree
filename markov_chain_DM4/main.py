import numpy as np


class MarkovChain(object):
    def __init__(self, transition_matrix, states):
        """
            @param transition_matrix: 2D array representing the probabilities of change of
            state in the Markov Chain.
            @param states: array representing the states of the Markov Chain. It
            needs to be in the same order as transition_matrix.
        """
        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.states = states
        self.index_dict = {self.states[index]: index for index in
                           range(len(self.states))}
        self.state_dict = {index: self.states[index] for index in
                           range(len(self.states))}

    def next_state(self, current_state):
        """
            @param current_state: current state of the system.
            @return: state of the random variable at the next time
        """
        return np.random.choice(
            self.states,
            p=self.transition_matrix[self.index_dict[current_state], :]
        )

    def generate_states(self, current_state, no=10):
        """
            @param current_state: state of the current random variable.
            @param no: number of future states to generate.
            @return: generated the next states of the system.
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states


transition_matrix = [[0.9, 0.075, 0.025],
                         [0.15,  0.8,  0.05],
                         [0.25,  0.25,  0.5]]

weather_chain = MarkovChain(transition_matrix=transition_matrix,
                                states=['Sunny', 'Rainy', 'Snowy'])


print(weather_chain.next_state(current_state='Sunny'))
print(weather_chain.next_state(current_state='Snowy'))
print(weather_chain.generate_states(current_state='Snowy', no=10))