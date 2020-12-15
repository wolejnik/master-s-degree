from igraph import *
import matplotlib.pyplot as plt
import igraph
import numpy as np


def badanie(g, size, n, b, m, a):
    # size = 30   # wielkość sieci
    # n = 10      # początkowa ilość zarażonych węzłów
    # b = 0.5     # prawdopodobieństwo zarażenia
    # m = 0.1     # prawdopodobieństwo wyzdrowienia
    # g = Graph.Watts_Strogatz(dim=1, size=size, nei=3, p=0.1)
    g.vs['infected'] = False
    g.vs['recovered'] = False
    g.vs['availability'] = False
    g.vs['color'] = 'blue'
    inf_list = []
    rec_list = []
    S = [size]
    I = [0]
    R = [0]

    patient_zero = np.random.choice(g.vs, n)
    for vertex in patient_zero:
        g.vs[vertex.index]['infected'] = True
        g.vs[vertex.index]['color'] = 'red'
        if vertex.index not in inf_list:
            inf_list.append(vertex.index)

    print(f'Zarażone węzły łącznie: {len(inf_list)}')
    print('---------------------------------')

    g_next = g
    i = 0


    S.append(size-len(inf_list))
    I.append(len(inf_list))
    R.append(len(rec_list))
    while True:
        i += 1
        for vertex in g.vs:
            if np.random.random() < a:
                vertex['availability'] = True
            else:
                vertex['availability'] = False
        for vertex in g.vs:
            if vertex['infected'] == True and vertex['recovered'] == False and vertex['availability'] == True:
                for vertex_new in vertex.neighbors():
                    if np.random.random() < b and vertex['availability'] == True:
                        g_next.vs[vertex_new.index]['infected'] = True
                        g_next.vs[vertex.index]['color'] = 'red'
                        if vertex_new.index not in inf_list:
                            inf_list.append(vertex_new.index)

        for vertex in g.vs:
            if vertex['infected'] == True and vertex['recovered'] == False:
                if np.random.random() < m:
                    print(f'OZDROWIAŁ WĘZEŁ: {vertex.index}')
                    g_next.vs[vertex.index]['recovered'] = True
                    g_next.vs[vertex.index]['infected'] = False
                    g_next.vs[vertex.index]['color'] = 'green'
                    if vertex.index not in rec_list:
                        rec_list.append(vertex.index)
        g = g_next

        print(f'Ozdrowiałe węzły: {len(rec_list)}')
        print(f'Zarażone węzły łącznie: {len(inf_list)}')
        print('---------------------------------')
        S.append(size - len(inf_list))
        I.append(len(inf_list)-len(rec_list))
        R.append(len(rec_list))
        # out = plot(g, layout='kk').save(f'{i}.png')
        if sorted(inf_list) == sorted(rec_list):
            break
    return S, I, R


if __name__ == "__main__":
    g = Graph.Watts_Strogatz(dim=1, size=100, nei=3, p=0.1)
    # bs = [0.1, 0.3]
    # ms = [0.1, 0.3]
    bs = [0.05]
    ms = [0.05]
    a = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    i = 231
    for val in a:
        for b in bs:
            for m in ms:
                S, I, R = badanie(g, 100, 20, b, m, val)
                # print(f'S: {S}')
                # print(f'I: {I}')
                # print(f'R: {R}')

                # plt.subplot(i)
                plt.plot(S)
                plt.plot(I)
                plt.plot(R)
                plt.legend(('S', 'I', 'R'))
                plt.title(f'PZ = {b}, PW = {m}, D = {val}')
                # b = 0.5     # prawdopodobieństwo zarażenia
                # m = 0.1     # prawdopodobieństwo wyzdrowienia
                i += 1
                plt.show()