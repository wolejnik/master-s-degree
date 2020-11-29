from igraph import *
import matplotlib.pyplot as plt
import igraph
import numpy as np

def licz(g):
    active = 0
    for vertex in g.vs:
        if vertex['active'] == True:
            vertex['color'] = 'red'
            active += 1
    return active

def siec(g, m, n):
    g.vs['active'] = False
    g.vs['color'] = 'blue'
    g.vs['label'] = [vertex.index for vertex in g.vs]
    active_zero = np.random.choice(g.vs, m)
    A = []
    for vertex in active_zero:
        g.vs[vertex.index]['active'] = True
        vertex['color'] = 'red'

    A.append(licz(g))
    g_next = g
    i = 0
    plot(g, layout='kk')
    while True:
        i += 1
        for vertex in g.vs:
            active_neighbors = 0
            if vertex['active'] == False:
                for neighbor in vertex.neighbors():
                    if neighbor['active']:
                        active_neighbors += 1

            if active_neighbors/len(vertex.neighbors()) >= n:
                print(f'Węzeł {vertex.index} ma {(active_neighbors / len(vertex.neighbors())) * 100}% aktywnych sąsiadów')
                g_next.vs[vertex.index]['active'] = True
                g_next.vs[vertex.index]['color'] = 'red'
        plot(g, layout='kk')
        A.append(licz(g_next))
        g = g_next
        if A[-1] == A[-2]:
            break

    return A

if __name__ == "__main__":
    g = Graph.Watts_Strogatz(dim=1, size=1000, nei=4, p=0.1)
    ms = [30, 100]
    ns = [0.3, 0.2]
    i = 221
    for m in ms:
        for n in ns:
            A = siec(g, m, n)
            print(A)
            plt.subplot(i)
            plt.plot(A)
            plt.legend(('A'))
            plt.title(f'm = {m/10}%, n = {n*100}%')
            i += 1
    plt.show()