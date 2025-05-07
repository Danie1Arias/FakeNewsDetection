# seir_model.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SEIRFakeNewsModel:
    def __init__(self, G, beta, delta, gamma, alpha, eta, initial_state=None):
        self.G = G
        self.N = G.number_of_nodes()
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.alpha = alpha
        self.eta = np.array(eta)
        self.states = {node: 'S' for node in G.nodes}
        if initial_state:
            self.states.update(initial_state)

    def step(self):
        new_states = self.states.copy()
        for i in self.G.nodes:
            if self.states[i] == 'S':
                for j in self.G.neighbors(i):
                    if self.states[j] == 'I' and np.random.rand() < self.beta:
                        new_states[i] = 'E'
                        break
            elif self.states[i] == 'E':
                if np.random.rand() < self.delta:
                    new_states[i] = 'R' if np.random.rand() < self.eta[i] else 'I'
            elif self.states[i] == 'I':
                if np.random.rand() < self.gamma:
                    new_states[i] = 'R' if np.random.rand() < self.alpha else 'S'
        self.states = new_states

    def run(self, Tmax):
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

def simulate_seir(G, beta, delta, gamma, alpha, eta, initial_state, Tmax=100):
    model = SEIRFakeNewsModel(G, beta, delta, gamma, alpha, eta, initial_state)
    return model.run(Tmax)

def plot_seir(history, title='SEIR Model'):
    S, E, I, R = [], [], [], []
    for step in history:
        S.append(step['S'])
        E.append(step['E'])
        I.append(step['I'])
        R.append(step['R'])

    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Susceptible (S)')
    plt.plot(E, label='Exposed (E)')
    plt.plot(I, label='Infected (I - Spreading fake)')
    plt.plot(R, label='Recovered (R)')
    plt.xlabel('Time step')
    plt.ylabel('Number of nodes')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
