#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ntpath
from tqdm import tqdm
from sklearn.neighbors import BallTree

# Wczytanie danych

data = np.loadtxt("wraki-utm.txt")
print(data)


# In[22]:


# Pobranie danych od uzytkownika
grid = float(input(f'Określ rozdzielczość grid: '))
print(grid)
radius_window = float(input(f'Określ rozmiar okienka (promień okręgu): '))
print(radius_window)
min_points = float(input(f'Określ minimalną liczbe punktów niezbędnych do wyliczenia nowego punktu: '))
print(min_points)
# grid = 0.2
# radius = 0.4
# minPoints = 3


# In[23]:


# Stworzenie siatki
def create_grid(data, resolution):
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    x = np.arange(min_values[0], max_values[0], resolution)
    y = np.arange(min_values[1], max_values[1], resolution)
    return np.meshgrid(x, y)


# In[24]:


xMeshGrid, yMeshGrid = create_grid(data, grid)
print(f'{xMeshGrid}\n')
print(f'{yMeshGrid}\n')


# In[27]:


# Interpolacja metodą Moving Average
def moving_average(data, xMesh, yMesh, min_points, radius_window):
    ballTree = BallTree(data[:, :2])
    zMesh = np.empty((xMeshGrid.shape[0], xMeshGrid.shape[1]))

    for i in tqdm(range(xMesh.shape[0]), desc='Loading...'):
        for j in range(xMesh.shape[1]):
            idx = ballTree.query_radius([[xMesh[i, j], yMesh[i, j]]], r=radius_window)

            if len(idx[0]) < min_points:
                zMesh[i, j] = np.NaN
            else:
                zMesh[i, j] = np.mean(data[idx[0], 2])
    return zMesh


# In[28]:


zMeshGrid = moving_average(data, xMeshGrid, yMeshGrid, min_points, radius_window)
print(f'{zMeshGrid}\n')


# In[12]:


# Wyświetlenie wyniku
plt.figure()
plt.title("wraki_samochodów")
p = plt.imshow(zMeshGrid)
plt.colorbar(p)
plt.show()


# In[10]:


# Zapis do pliku CSV
data = {'X': xMeshGrid.flatten(), 'Y': yMeshGrid.flatten(), 'Z': zMeshGrid.flatten()}
df = pd.DataFrame(data)
df.to_csv("wraki_samochodów_outout", header=False, sep=' ', na_rep='NaN')

