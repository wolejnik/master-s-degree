from igraph import *
import matplotlib.pyplot as plt
import igraph
import numpy as np


def badanie_kto_pierwszy_ten_lepszy(g, size, n, b1, b2, m):
    # size = 30   # wielkość sieci
    # n = 10      # początkowa ilość zarażonych węzłów
    # b = 0.5     # prawdopodobieństwo zarażenia
    # m = 0.1     # prawdopodobieństwo wyzdrowienia
    # g = Graph.Watts_Strogatz(dim=1, size=size, nei=3, p=0.1)
    g.vs['infected'] = False
    g.vs['recovered'] = False
    g.vs['color'] = 'blue'
    g.vs['virus1'] = False
    g.vs['virus2'] = False
    inf_list = []
    inf1_list = []
    inf2_list = []
    rec_list = []
    rec1_list = []
    rec2_list = []
    S = [size]
    I = [0]
    R = [0]
    I1 = [0]
    I2 = [0]
    R1 = [0]
    R2 = [0]

    patient_zero = np.random.choice(g.vs, n*2, replace=False)
    for i, vertex in enumerate(patient_zero):
        if i < n:
            g.vs[vertex.index]['infected'] = True
            g.vs[vertex.index]['virus1'] = True
            g.vs[vertex.index]['color'] = 'pink'
            if vertex.index not in inf1_list:
                inf1_list.append(vertex.index)
        else:
            g.vs[vertex.index]['infected'] = True
            g.vs[vertex.index]['virus2'] = True
            g.vs[vertex.index]['color'] = 'red'
            if vertex.index not in inf2_list:
                inf2_list.append(vertex.index)
        if vertex.index not in inf_list:
            inf_list.append(vertex.index)

    print(f'Zarażone węzły łącznie: {len(inf_list)}')
    print('---------------------------------')

    g_next = g
    i = 0


    S.append(size-len(inf_list))
    I.append(len(inf_list))
    R.append(len(rec_list))
    I1.append(len(inf1_list))
    I2.append(len(inf2_list))
    while True:
        i += 1
        for vertex in g.vs:
            if vertex['recovered'] == False and vertex['infected'] == False:
                virus1_neighbors = 0
                virus2_neighbors = 0
                for neighbor in vertex.neighbors():
                    if neighbor['virus1']:
                        virus1_neighbors += 1
                    if neighbor['virus2']:
                        virus2_neighbors += 1
                # print('dupa')
                if virus1_neighbors != 0 or virus2_neighbors != 0:
                    if b1 * virus1_neighbors > b2 * virus2_neighbors:
                        if np.random.random() < b1:
                            print('zarazam v1')
                            g_next.vs[vertex.index]['infected'] = True
                            g_next.vs[vertex.index]['virus1'] = True
                            g.vs[vertex.index]['color'] = 'pink'
                            if vertex.index not in inf1_list:
                                inf1_list.append(vertex.index)
                            if vertex.index not in inf_list:
                                inf_list.append(vertex.index)
                    if b1 * virus1_neighbors < b2 * virus2_neighbors:
                        if np.random.random() < b2:
                            print('zarazam v2')
                            g_next.vs[vertex.index]['infected'] = True
                            g_next.vs[vertex.index]['virus2'] = True
                            g.vs[vertex.index]['color'] = 'red'
                            if vertex.index not in inf2_list:
                                inf2_list.append(vertex.index)
                            if vertex.index not in inf_list:
                                inf_list.append(vertex.index)
        for vertex in g.vs:
            if vertex['infected'] == True and vertex['recovered'] == False:
                if np.random.random() < m:
                    print(f'OZDROWIAŁ WĘZEŁ: {vertex.index}')
                    g_next.vs[vertex.index]['recovered'] = True
                    g_next.vs[vertex.index]['infected'] = False
                    g_next.vs[vertex.index]['color'] = 'green'
                    if vertex.index not in rec_list:
                        rec_list.append(vertex.index)
                    if vertex['virus1']:
                        if vertex.index not in rec1_list:
                            rec1_list.append(vertex.index)
                    if vertex['virus2']:
                        if vertex.index not in rec2_list:
                            rec2_list.append(vertex.index)
        g = g_next

        print(f'Ozdrowiałe węzły: {len(rec_list)}')
        print(f'Zarażone węzły łącznie: {len(inf_list)}')
        print('---------------------------------')
        S.append(size - len(inf_list))
        I.append(len(inf_list)-len(rec_list))
        R.append(len(rec_list))
        I1.append(len(inf1_list)-len(rec1_list))
        I2.append(len(inf2_list)-len(rec2_list))
        R1.append(len(rec1_list))
        R2.append(len(rec2_list))
        # g.vs['color'] = ['green' if vertex['recovered'] == True else 'blue' if vertex['infected'] == False else 'red' for vertex in g.vs]
        # out = plot(g, layout='kk').save(f'{i}.png')
        if sorted(inf_list) == sorted(rec_list):
            break
    return S, I, R, I1, I2, R1, R2

def badanie_ciagla_rywalizacja(g, size, n, b1, b2, m):
    # size = 30   # wielkość sieci
    # n = 10      # początkowa ilość zarażonych węzłów
    # b = 0.5     # prawdopodobieństwo zarażenia
    # m = 0.1     # prawdopodobieństwo wyzdrowienia
    # g = Graph.Watts_Strogatz(dim=1, size=size, nei=3, p=0.1)
    g.vs['infected'] = False
    g.vs['recovered'] = False
    g.vs['color'] = 'blue'
    g.vs['virus1'] = False
    g.vs['virus2'] = False
    inf_list = []
    inf1_list = []
    inf2_list = []
    rec_list = []
    rec1_list = []
    rec2_list = []
    S = [size]
    I = [0]
    R = [0]
    I1 = [0]
    I2 = [0]
    R1 = [0]
    R2 = [0]

    patient_zero = np.random.choice(g.vs, n*2, replace=False)
    for i, vertex in enumerate(patient_zero):
        if i < n:
            g.vs[vertex.index]['infected'] = True
            g.vs[vertex.index]['virus1'] = True
            g.vs[vertex.index]['color'] = 'pink'
            if vertex.index not in inf1_list:
                inf1_list.append(vertex.index)
        else:
            g.vs[vertex.index]['infected'] = True
            g.vs[vertex.index]['virus2'] = True
            g.vs[vertex.index]['color'] = 'red'
            if vertex.index not in inf2_list:
                inf2_list.append(vertex.index)
        if vertex.index not in inf_list:
            inf_list.append(vertex.index)

    print(f'Zarażone węzły łącznie: {len(inf_list)}')
    print('---------------------------------')

    g_next = g
    i = 0


    S.append(size-len(inf_list))
    I.append(len(inf_list))
    R.append(len(rec_list))
    I1.append(len(inf1_list))
    I2.append(len(inf2_list))
    while True:
        i += 1
        for vertex in g.vs:
            if vertex['recovered'] == False:
                virus1_neighbors = 0
                virus2_neighbors = 0
                for neighbor in vertex.neighbors():
                    if neighbor['virus1']:
                        virus1_neighbors += 1
                    if neighbor['virus2']:
                        virus2_neighbors += 1
                # print('dupa')
                if virus1_neighbors != 0 or virus2_neighbors != 0:
                    if b1 * virus1_neighbors > b2 * virus2_neighbors:
                        if np.random.random() < b1 and not vertex['virus1']:
                            g_next.vs[vertex.index]['infected'] = True
                            g_next.vs[vertex.index]['virus1'] = True
                            g_next.vs[vertex.index]['virus2'] = False
                            g.vs[vertex.index]['color'] = 'pink'
                            if vertex.index not in inf1_list:
                                inf1_list.append(vertex.index)
                            if vertex.index in inf2_list:
                                inf2_list.pop(inf2_list.index(vertex.index))
                            if vertex.index not in inf_list:
                                inf_list.append(vertex.index)
                    if b1 * virus1_neighbors < b2 * virus2_neighbors:
                        if np.random.random() < b2 and not vertex['virus2']:
                            g_next.vs[vertex.index]['infected'] = True
                            g_next.vs[vertex.index]['virus1'] = False
                            g_next.vs[vertex.index]['virus2'] = True
                            g.vs[vertex.index]['color'] = 'red'
                            if vertex.index not in inf2_list:
                                inf2_list.append(vertex.index)
                            if vertex.index in inf1_list:
                                inf1_list.pop(inf1_list.index(vertex.index))
                            if vertex.index not in inf_list:
                                inf_list.append(vertex.index)
        # out = plot(g_next, layout='kk').save(f'{i}.png')
        for vertex in g.vs:
            if vertex['infected'] == True and vertex['recovered'] == False:
                if np.random.random() < m:
                    print(f'OZDROWIAŁ WĘZEŁ: {vertex.index}')
                    g_next.vs[vertex.index]['recovered'] = True
                    g_next.vs[vertex.index]['infected'] = False
                    g_next.vs[vertex.index]['color'] = 'green'
                    if vertex.index not in rec_list:
                        rec_list.append(vertex.index)
                    if vertex['virus1']:
                        if vertex.index not in rec1_list:
                            rec1_list.append(vertex.index)
                    if vertex['virus2']:
                        if vertex.index not in rec2_list:
                            rec2_list.append(vertex.index)

        # out = plot(g_next, layout='kk').save(f'{i}.png')
        g = g_next

        print(f'Ozdrowiałe węzły: {len(rec_list)}')
        print(f'Zarażone węzły łącznie: {len(inf_list)}')
        print('---------------------------------')
        S.append(size - len(inf_list))
        I.append(len(inf_list)-len(rec_list))
        R.append(len(rec_list))
        I1.append(len(inf1_list)-len(rec1_list))
        I2.append(len(inf2_list)-len(rec2_list))
        R1.append(len(rec1_list))
        R2.append(len(rec2_list))
        # g.vs['color'] = ['green' if vertex['recovered'] == True else 'blue' if vertex['infected'] == False else 'red' for vertex in g.vs]
        # out = plot(g, layout='kk').save(f'{i}.png')
        if sorted(inf_list) == sorted(rec_list):
            break
    return S, I, R, I1, I2, R1, R2


if __name__ == "__main__":
    g = Graph.Watts_Strogatz(dim=1, size=100, nei=3, p=0.1)
    # bs = [0.1, 0.3]
    # ms = [0.1, 0.3]
    bs = [0.1, 0.2]
    ms = [0.1]
    # j = 221
    # for b1 in bs:
    #     for b2 in bs:
    #         for m in ms:
    #             S, I, R, I1, I2, R1, R2 = badanie_kto_pierwszy_ten_lepszy(g, 100, 20, 0.1, 0.1, m)
    #             # S, I, R, I1, I2, R1, R2 = badanie_ciagla_rywalizacja(g, 100, 20, 0.1, 0.1, m)
    #             # print(f'S: {S}')
    #             # print(f'I: {I}')
    #             # print(f'I1: {I1}')
    #             # print(f'I2: {I2}')
    #             # print(f'R: {R}')
    #             # print(f'R1: {R1}')
    #             # print(f'R2: {R2}')
    #
    #             # plt.subplot(j)
    #             plt.plot(S)
    #             plt.plot(I)
    #             plt.plot(R)
    #             plt.plot(I1)
    #             plt.plot(I2)
    #             # plt.legend(('S', 'I', 'R'))
    #             plt.legend(('S', 'I', 'R', 'virus 1', 'virus 2'))
    #             plt.title(f'v1 = {b1}, v2 = {b2}, m = {m}')
    #             j += 1
    #             plt.show()
    # plt.show()

    j = 221
    for b1 in bs:
        for b2 in bs:
            for m in ms:
                S, I, R, I1, I2, R1, R2 = badanie_ciagla_rywalizacja(g, 100, 20, 0.1, 0.1, m)
                # print(f'S: {S}')
                # print(f'I: {I}')
                # print(f'I1: {I1}')
                # print(f'I2: {I2}')
                # print(f'R: {R}')
                # print(f'R1: {R1}')
                # print(f'R2: {R2}')

                # plt.subplot(j)
                plt.plot(S)
                plt.plot(I)
                plt.plot(R)
                plt.plot(I1)
                plt.plot(I2)
                # plt.legend(('S', 'I', 'R'))
                plt.legend(('S', 'I', 'R', 'virus 1', 'virus 2'))
                plt.title(f'v1 = {b1}, v2 = {b2}, m = {m}')
                j += 1
                plt.show()