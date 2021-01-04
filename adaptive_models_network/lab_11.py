import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import random
import copy
import cv2


def symulacja(network, n, b, m, usuniecie, chore_indeksy):
    step = 0
    zdrowe_wezly = []
    chore_wezly = []
    odporne_wezly = []
    warunek_stopu = True

    indeksy = [node.index for node in network.vs]

    network.vs['infected'] = 0
    network.vs['recovered'] = 0
    network.vs['deleted'] = False
    network.vs['color'] = 'green'

    network.vs[chore_indeksy]['infected'] = 1
    network.vs[chore_indeksy]['color'] = 'red'

    usuniete_indeksy = random.sample(indeksy, k=int(usuniecie * len(network.vs) / 100))
    network.vs[usuniete_indeksy]['deleted'] = True
    network.vs[usuniete_indeksy]['color'] = 'black'

    ig.plot(network, layout="kamada_kawai").save("obrazek.png")
    img=cv2.imread("obrazek.png")

    plt.imshow(img)
    plt.title(str("zadanie 1, początkowa symulacja, deleted = " + str(usuniecie) + "%"))
    plt.show()

    while warunek_stopu:
        network2 = copy.deepcopy(network)
        step += 1
        zdrowe_suma = 0
        chore_suma = 0
        odporne_suma = 0

        for node in network.vs:
            if (not node['deleted']):
                chore_suma += node['infected']
                odporne_suma += node['recovered']
                zdrowe_suma = int(len(network.vs) * (100 - usuniecie) / 100) - chore_suma - odporne_suma

                infected_neighbors_sum = 0
                for neighbor in node.neighbors():
                    if (neighbor['infected'] == 1 and not neighbor['deleted']):
                        infected_neighbors_sum += 1
                        break

                if (infected_neighbors_sum and node[
                    'recovered'] != 1 and np.random.rand() * infected_neighbors_sum < b):
                    network2.vs[node.index]['infected'] = 1
                    network2.vs[node.index]['color'] = 'red'

                if (node['infected'] == 1 and np.random.rand() * 100 < m):
                    network2.vs[node.index]['infected'] = 0
                    network2.vs[node.index]['recovered'] = 1
                    network2.vs[node.index]['color'] = 'blue'

        network = copy.deepcopy(network2)
        zdrowe_wezly.append(zdrowe_suma)
        chore_wezly.append(chore_suma)
        odporne_wezly.append(odporne_suma)
        if (chore_suma == 0): warunek_stopu = False

    f = open(str("zad1_usuniecie" + str(usuniecie) + ".txt"), 'w')
    f.write(str("Zdrowe: "))
    for i in range(len(zdrowe_wezly)): f.write(str(str(zdrowe_wezly[i]) + ", "))
    f.write(str("\nChore: "))
    for i in range(len(chore_wezly)): f.write(str(str(chore_wezly[i]) + ", "))
    f.write(str("\nOdporne: "))
    for i in range(len(odporne_wezly)): f.write(str(str(odporne_wezly[i]) + ", "))
    f.close()

    plt.plot(np.arange(len(zdrowe_wezly)), zdrowe_wezly, label='Zdrowe (S)')
    plt.plot(np.arange(len(chore_wezly)), chore_wezly, label='Chore (I)')
    plt.plot(np.arange(len(odporne_wezly)), odporne_wezly, label='Odporne (R)')
    plt.xlabel('Iteracje')
    plt.ylabel('Ilosc wezlow')
    plt.legend()
    plt.grid()
    plt.title("zadanie 1, usuniecie=" + str(usuniecie) + "%")
    plt.show()


# Zadanie 1

network = ig.Graph.Watts_Strogatz(dim=1, size=500, nei=1, p=0.5)
indeksy = [node.index for node in network.vs]
n=5
chore_indeksy = random.sample(indeksy, k = int(n*len(network.vs)/100))

for usuniecie in [0,5,10,20]:
    symulacja(network, n=5, b=5, m=15, usuniecie=usuniecie, chore_indeksy=chore_indeksy)