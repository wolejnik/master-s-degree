from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt

# Zadanie 1
def interpoluj_kNN(xi, x, y, k):
    """
    @param xi: wartosc, na ktore chcemy dokonac interpolacji
    @param x: macierz argumentow wezlow interpolacji
    @param y: wektor wartosci wezlow interpolacji
    @param k: liczba uwzglednianych sasiadow
    @return yi: srednia arytmetyczna z wartosci y dla k najblizszych sasiadow
    """

    pass

# Zadanie 1
def interpoluj_kNN(x, xc, yc, k):
    """

    @param x: wartość, dla ktorej chcemy dokonac klasyfikacji
    @param xc: macierz atrybutow wejsciowych danych uczacych
    @param yc: wektor klas zdanych
    @param k: liczba uwzglednianych sasiadow 
    @return res : najczestsza klase sposrod k najbliszych sasiadow
    """

    pass

if __name__ == "__main__":
    print()


    # tree = spatial.KDTree(xc)
    # distances_neig, indexes_neig = tree.query(x, k)
    # unique_neig, unique_indexes_neig = np.unique(y[indexes_neig], return_counts=True)
    # max_unique_indexes_neig = np.argmax(unique_indexes_neig)
    # return unique_neig[max_unique_indexes_neig]