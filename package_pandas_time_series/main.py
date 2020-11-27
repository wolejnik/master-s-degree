import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

def main():
    """ Main program """

    # Zadanie 1. Wczytaj dane z pliku stock.arff. W poszczególnych kolumnach pliku znajdują
    # się notowania giełdowe 10 spółek od 1 stycznia 1988 roku (950 notowań z każdego
    # kolejnego dnia).
    data, meta = arff.loadarff('stock.arff')

    # Zadanie 2. Na podstawie wczytanych danych utwórz obiekt DataFrame indeksowany kolejnymi
    # datami notowania.
    df = pd.DataFrame(data)

    # Zadanie 3. Sporządź wykresy obrazujące przebieg notowań w czasie.
    df.index = pd.date_range(start='1/1/1988', periods=len(df), freq='D')
    df.plot()
    plt.savefig('chart_1.png')


    # Zadanie 4.Uśrednij dane oknem czasowym o założonym rozmiarze i sporządź wykresy danych
    # uśrednionych.
    df2 = df.rolling(100, min_periods=1).mean()
    df2.plot()
    plt.show()

    plt.savefig('chart_2.png')


    return 0

if __name__ == "__main__":
    main()