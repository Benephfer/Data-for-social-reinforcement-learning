import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

#from numba import njit, prange
from numba import njit
#np.random.seed(seed = 42)


def spatial_random_graph(N, r):
    pos = np.random.rand(N, 2)  # random positions in [0,1]x[0,1]
    coordinate_dict = {}

# Iterate through the indices and corresponding coordinates
    for i, coord in enumerate(pos):
        coordinate_dict[i] = tuple(coord)
           
    linkList = np.zeros((N, N + 1), dtype=int)  # empty array for neighbors

    # go through all possible pairs and check if linked
    for a1 in range(N):
        linkList[a1, 0] = 0
        for a2 in range(N):
            # euclidean distance smaller than r (and a1 != a2)?
            dist = np.sqrt((pos[a1, 0] - pos[a2, 0])**2 + (pos[a1, 1] - pos[a2, 1])**2)
            if a1 != a2 and dist < r:
                # count number of neighbors in the first element of linkList
                linkList[a1, 0] += 1
                # add neighbor
                linkList[a1, linkList[a1, 0]] = a2

    return linkList, pos, coordinate_dict
@njit
def update_opinions(Q, a1, expression, linkList, alpha, preference):
    numN = linkList[a1, 0]
    a2 = linkList[a1, np.random.randint(numN) + 1]
    reaction = int(Q[a2, 1] > Q[a2, 0])  # response opinion articulated by a2
    reward = (expression * 2 - 1) * (reaction * 2 - 1)  # agreement -> 1 | disagreement -> -1

    # update of Q-values for a1 considering what has been expressed
    Q[a1, expression] = (1 - alpha) * Q[a1, expression] + alpha * reward
    preference[a1] = int(Q[a1, 1] > Q[a1, 0])
    

def plot_opinions(pos, Q, r, N, coordinates, t, pref):
    plt.figure(figsize=(4, 4))
    G = nx.random_geometric_graph(N, r, pos = coordinates)
    nx.draw_networkx(G, pos = pos, node_size = 0, with_labels = False)
    plt.scatter(pos[:, 0], pos[:, 1], vmin=0, vmax=1, c=Q[:, 1] > Q[:, 0], cmap='coolwarm', s=75, edgecolors='k')
    #plt.scatter(pos[:, 0], pos[:, 1], c=pref[:][0], cmap='coolwarm', s=150, edgecolors='k')
    plt.title('Agent Opinions')
    plt.xlabel('x-position')
    plt.ylabel('y-position')
    plt.colorbar(label='Opinion: 1 > -1')
    plt.tight_layout()
    plt.savefig('opinions_t{}.png'.format(t), dpi=600)
    plt.clf()
    plt.cla()
   
#@njit
def simulate(steps, population, Qvals, preferences, network, alpha, eps, printstep):
    ctime = 0
    pref_t = []
    for step in range(steps):
    #while True:
        a1 = np.random.randint(population)  # random choice of agent that expresses his opinion
        expression = int(Qvals[a1, 1] > Qvals[a1, 0])  # expression (1 -> o = 1 | 0 -> o = -1)

        if np.random.rand() < eps:
            expression = -(expression - 1)  # exploration

        update_opinions(Qvals, a1, expression, network, alpha, preferences)  
        total = np.sum(preferences)
        pref_t.append(total)
        if step in printstep:
            plot_opinions(pos, Qvals, r ,N, coords, step, pref_t)
        if total == 0 or total == N:
            ctime = step
            #print(ctime)
            plot_opinions(pos, Qvals, r ,N, coords, step, pref_t)
            break
    #print(ctime)    
    return ctime, pref_t


# Parameters
N = 100
r = 0.25
steps = 1000*1000*1000*10
alpha = 0.25
eps = 0.1
stops = []
printstep = [1, 50, 100, 1000, 10*1000, 100*1000, 200*1000, 300*1000, 1000*1000, 1000*1000*10, 1000*1000*100, 1000*1000*1000]

# Generate spatial random graph
check = True
while check:
    
    linkList, pos, coords = spatial_random_graph(N, r)
    G = nx.random_geometric_graph(N, r, pos = coords)
    if nx.is_connected(G):
        check = False

# Random initial values in [-0.5, 0.5] for both opinions
Q = np.random.rand(N, 2) - 0.5
pref = [int(Q[a, 1] > Q[a, 0]) for a in range(0,N)]
pref = np.array(pref)

stoptime, totals = simulate(steps = steps, population = N, Qvals = Q, preferences = pref, network = linkList, alpha = alpha, eps = eps, printstep = printstep)

plt.figure(figsize = (5,3))
plt.plot(totals, color = 'k')
plt.grid()
plt.ylim(0,100)
plt.xlabel('$t$')
plt.ylabel('# agents with $O=1$')
plt.tight_layout()
plt.savefig('totals.png', dpi = 600)

pd.DataFrame(totals).to_csv('totals.csv', header = None, index = None)
#stops.append(stoptime)

#pd.DataFrame(stops).to_csv('stops_N{}_r{}_a{}.csv'.format(N,int(r*100),int(100*alpha)), header = None, index = None)

