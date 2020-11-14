import numpy as np
import pandas as pd

# Zadanie 1

"""
Utwórz obiekt DataFrame z czterema kolumnami o nazwach: ’a’, ’b’, ’c’, ’d’ i tysią-
cem wierszy, wypełniony liczbami losowymi o rozkładzie normalnym N (0, 1).
"""

columns = ['a', 'b', 'c', 'd']
df = pd.DataFrame(dict(a=np.random.normal(0, 1, size=1000),
                       b=np.random.normal(0, 1, size=1000),
                       c=np.random.normal(0, 1, size=1000),
                       d=np.random.normal(0, 1, size=1000)),
                  columns=columns)

print(df)

# Zadanie 1.1

# Zadanie 1.2

# Zadanie 1.3

# Zadanie 1.4