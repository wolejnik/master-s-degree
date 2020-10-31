import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris(as_frame=True)
X = iris.data
Y = iris.target
irisX = pd.DataFrame(X)
irisY = pd.DataFrame(Y)
print("---> Zadanie 1 <---")
print(irisX.dtypes)
print(irisY.dtypes)
print("-------")

#zad_2

dataZoo = pd.read_csv('zoo.csv')
dataZoo1 = pd.DataFrame(dataZoo)
dataAutos = pd.read_csv('autos.csv')
dataAutos1 = pd.DataFrame(dataAutos)
df_ir = pd.DataFrame(iris.data, columns=iris.feature_names)

print("---> Zadanie 2 <---")
print(dataZoo.dtypes)
print("-------")
print(dataAutos.dtypes)
print("-------")
print(df_ir.dtypes)
print("-------")

#zad_3

booleansZoo = dataZoo.select_dtypes(include=bool)
booleansAutos = dataAutos.select_dtypes(include=bool)
booleansIris = df_ir.select_dtypes(include=bool)

print(booleansZoo.apply(pd.Series.value_counts))
print(booleansAutos.apply(pd.Series.value_counts))
print(booleansIris.apply(pd.Series.value_counts))

# Średnie
zoo2 = dataZoo.select_dtypes(exclude = "bool")
print(zoo2.mean())

auto2 = dataAutos.select_dtypes(exclude = "bool")
print(auto2.mean())

df_ir2 = df_ir.select_dtypes(exclude = "bool")
print(df_ir2.mean())
# Odchylenie standardowe
print(zoo2.std())
print(auto2.std())
print(df_ir2.std())

# Zadanie czwarte
zoo2.hist()
plt.show()
zoo2.boxplot()

irisX.hist(edgecolor='black')
plt.suptitle('Iris')
plt.show()

irisX.boxplot()
plt.title('Iris')
plt.show()

auto2.hist()
plt.show()
auto2.boxplot()

df_ir2.hist()
plt.show()
plt.title('Autos')
df_ir2.boxplot()

#zad5
pca = PCA(n_components=3)
pca.fit(iris.data)
pca.explained_variance_ratio_

#zad6

def zad6(variance_ratio):
    amount = 0
    components = []
    for index, val in enumerate(variance_ratio):
        amount += val
        components.append(index)
        if amount > 0.9:
            return components

print(f'Variance ratio:  {pca.explained_variance_ratio_}')
selected_components = zad6(pca.explained_variance_ratio_)
print(f'Selected components : {selected_components}')

#zadanie 7

#2D

plt.figure()
X_r = pca.fit(X).transform(X)
target_names = iris.target_names
for color, i, target_name in zip(['navy', 'turquoise', 'darkorange'], [0, 1, 2], target_names):
    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.8, lw=3,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.show()

#3D

X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

# Zadanie 8

# fig, ax = plt.subplots()
# ax.scatter(iris.data[:, 0], iris.data[:, 1], cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_xlabel('Sepal Lenght')
# ax.set_ylabel('Sepal Width')
