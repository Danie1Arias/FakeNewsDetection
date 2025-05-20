import numpy as np
import matplotlib.pyplot as plt

class SEIRFakeNewsModel:
    def __init__(self, G, beta, delta, mu, gamma, eta, initial_state=None):
        """
        G: networkx graph
        beta: infection/contact rate
        delta: decision rate (1/δ = avg. time to decide)
        mu: probability of an I node becoming R
        gamma: probability of an R node reverting to I
        eta: array of length N (per node probability of not spreading fake news)
        """
        self.G = G
        self.N = G.number_of_nodes()
        self.beta = beta
        self.delta = delta
        self.mu = mu
        self.gamma = gamma
        self.eta = np.array(eta)
        self.states = {node: 'S' for node in G.nodes}
        if initial_state:
            self.states.update(initial_state)

    def step(self):
        new_states = self.states.copy()
        for i in self.G.nodes:
            state = self.states[i]
            if state == 'S':
                if any(self.states[j] == 'I' and np.random.rand() < self.beta for j in self.G.neighbors(i)):
                    new_states[i] = 'E'
            elif state == 'E':
                if np.random.rand() < self.delta:
                    new_states[i] = 'R' if np.random.rand() < self.eta[i] else 'I'
            elif state == 'I':
                if np.random.rand() < self.mu:
                    new_states[i] = 'R'
            elif state == 'R':
                if np.random.rand() < self.gamma:
                    new_states[i] = 'I'
        self.states = new_states


    def run(self, Tmax=100):
        history = []
        for _ in range(Tmax):
            counts = self.count_states()
            history.append(counts)
            self.step()
        return history

    def count_states(self):
        count = {'S': 0, 'E': 0, 'I': 0, 'R': 0}
        for state in self.states.values():
            count[state] += 1
        return count


def simulate_seir(G, beta, delta, mu, gamma, eta, initial_state, Tmax=100):
    model = SEIRFakeNewsModel(G, beta, delta, mu, gamma, eta, initial_state)
    return model.run(Tmax)


def plot_seir(history, title='SEIR Simulation'):
    S, E, I, R = [], [], [], []
    for step in history:
        S.append(step['S'])
        E.append(step['E'])
        I.append(step['I'])
        R.append(step['R'])
    
    plt.figure(figsize=(10,6))
    plt.plot(S, label='Susceptible (S)')
    plt.plot(E, label='Exposed (E)')
    plt.plot(I, label='Infected (I - Spreading)')
    plt.plot(R, label='Removed (R)')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Number of Nodes")
    plt.grid(True)
    plt.legend()
    plt.show()
