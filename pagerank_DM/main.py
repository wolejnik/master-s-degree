import numpy as np

d = 0.9

P_dict = {
    'A': ['B', 'C', 'D'],
    'B': ['C'],
    'C': ['B', 'D'],
    'D': ['C'],
    'E': ['B', 'C', 'D']
}

P_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

A = np.zeros((len(P_dict), len(P_dict)))

for i, key in enumerate(P_dict):
    for v in P_dict[key]:
        index = list(P_index.keys())[list(P_index.values()).index(i)]
        A[i, P_index[v]] = 1/len(P_dict[index])

print(A)

E = np.ones((len(P_dict), len(P_dict)))

PR_1 = {
    'A': 1,
    'B': 1,
    'C': 1,
    'D': 1,
    'E': 1
}

for i in range(0, 20):
    for key in P_dict:
        temp = 0
        i = 0
        for v in P_dict.values():
            for x in v:
                if key == x:
                    index = list(P_index.keys())[list(P_index.values()).index(i)]
                    temp += PR_1[index]/len(P_dict[index])
            i += 1
        temp = (1-d) + d * temp
        PR_1[key] = round(temp, 3)

    print(f'Iteracja {i} metoda 1: {PR_1}')
