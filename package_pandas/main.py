import numpy as np
import pandas as pd

# Zadanie 1

"""
Utwórz obiekt DataFrame z czterema kolumnami o nazwach: ’a’, ’b’, ’c’, ’d’ i tysią-
cem wierszy, wypełniony liczbami losowymi o rozkładzie normalnym N (0, 1).
"""

df = pd.DataFrame({'$a': np.random.normal(0, 1, size=1000),
                   '$b': np.random.normal(0, 1, size=1000),
                   "$c": np.random.normal(0, 1, size=1000),
                   "$d": np.random.normal(0, 1, size=1000)})
df.columns = ['a', 'b', 'c', 'd']

# print(df)

# Zadanie 1.1
""" określą liczbę wszystkich elementów o wartościach spoza przedziału [−2, 2]; """

count_items_column = df[(df < -2) | (df > 2)].count()
count_items_all = count_items_column.sum()
print(f'\nLiczba elementów o wartosciach spoza przedziału [-2, 2]: {count_items_all}\n')

# Zadanie 1.2
""" określą liczbę elementów o wartościach spoza przedziału [−2, 2] w każdej kolumnie; """

print(f'Liczba elementów o wartosciach spoza przedziału [-2, 2] dla każdej kolumny :\n{count_items_column}\n')

# Zadanie 1.3
""" usuną wszystkie wiersze, w których występuje przynajmniej jeden element spoza przedziału [−2, 2]; """

deleted_items = df[(df >= -2) & (df <= 2)].dropna()
print(f'Obiekt z usuniętymi wierszami, w których wystęouje przynajmniej jeden element spoza przedziału [-2, 2]:\n{deleted_items}\n')

# Zadanie 1.4
""" zamienią wszystkie elementy ujemne na kwadrat ich wartości; 
    https://www.w3resource.com/pandas/dataframe/dataframe-applymap.php
"""

non_negative_items = df.applymap(lambda x: x**2 if x < 0 else x)
print(f'Obiekt z zamienionymi elementami ujemnymi na kwadrat ich wartości:\n{non_negative_items}\n')