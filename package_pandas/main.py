import numpy as np
import pandas as pd
import pandas as pd
from io import StringIO
from scipy.io import arff

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


# Zadanie 2

data, meta = arff.loadarff(StringIO('supermarket.arff'))
df1 = pd.DataFrame(data).select_dtypes([np.object]).stack().str.decode('utf-8').unstack()

# Zadanie 2.1
""" Określ ile jest próbek w pliku i ile z nich posiada brakujące wartości """

print(f'Liczba próbek w pliku:\n{df1.shape[0]}\n')
print(f'Liczba próbek z brakującymi wartościami:\n{df1.shape[0] - df1[df1 != "?"].dropna().shape[0]}\n')

# Zadanie 2.2
""" Określ ile próbek posiada 1 brakującą wartość, 2 brakujące wartości, . . . N
brakujących wartości (gdzie N to ilość atrybutów) """

df_missing = np.vstack((np.unique((df1=='?').sum(axis=1), return_counts=True)))
for i in range(df_missing.shape[1]):
    print(f'W próbce numer :{df_missing[1, i]} - brakuje {df_missing[0, i]} próbek.\n')


# Zadanie 2.3
""" Określ w ilu kolumnach występują brakujące wartości. Wypisz ich nazwy i liczby braków"""

df_missing_values = df1[df1 == '?'].count()
print(f'Ilość kolumn z brakujacymi wartosciami:\n{df_missing_values[df_missing_values>0].count()}\n')
print(f'Nazwy kolumn i ilość brakujących wartości:\n{df_missing_values[df_missing_values > 0]}\n')