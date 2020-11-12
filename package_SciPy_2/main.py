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

    tree = spatial.KDTree(x)
    distances_neig, indexes_neig = tree.query(xi, k)
    yi = 0
    for i in indexes_neig:
        yi += y[i]
    yi = yi / len(indexes_neig)
    return yi

    pass

# Zadanie 2
def klasyfikuj_kNN(x, xc, yc, k):
    """

    @param x: wartosc, dla ktorej chcemy dokonac klasyfikacji
    @param xc: macierz atrybutow wejsciowych danych uczacych
    @param yc: wektor klas zdanych
    @param k: liczba uwzglednianych sasiadow 
    @return res : najczestsza klase sposrod k najbliszych sasiadow
    """

    tree = spatial.KDTree(xc)
    distances_neig, indexes_neig = tree.query(x, k)
    unique_neig, unique_indexes_neig = np.unique(yc[indexes_neig], return_counts=True)
    max_unique_indexes_neig = np.argmax(unique_indexes_neig)
    return unique_neig[max_unique_indexes_neig]

# Zadanie 3
def walidacja(X, Y):
    """

    @param X:  zbior wartosci
    @param Y:  zbior klas
    @return best_k: optymalna liczba sasiadow
    @return best_score: srednia arytmetyczna z wartosci y dla k najblizszych sasiadow
    """
    best_score = 0.0
    best_k = 0
    for k in range(1, X.shape[0]//2):
        score = 0
        for i in range(X.shape[0]):
            trainX = np.concatenate((X[:i, :], X[i + 1:, :]), axis=0)
            trainY = np.concatenate((Y[:i], Y[i + 1:]), axis=0)
            testX = X[i, :]
            testY = Y[i]
            predicted = klasyfikuj_kNN(testX, trainX, trainY, k)
            if predicted == testY:
                score += 1
        score /= X.shape[0]
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score


if __name__ == "__main__":

    # Zadanie 1

    x = np.loadtxt('Dane/dane_3D_kapitan_i.txt')
    y = np.loadtxt('Dane/dane_3D_kapitan_o.txt')
    yi = interpoluj_kNN(x[0], x, y, 12)
    print("Wynik interpolacji: " + str(yi))

    # Zadanie 2

    xc = np.loadtxt('Dane/dane_2D_1_i.txt')
    yc = np.loadtxt('Dane/dane_2D_1_o.txt')
    my_class = klasyfikuj_kNN(xc[0], xc, yc, 8)
    print("Wynik klasyfikacji: " + str(yi))


    # Zadanie 3

    # x3 = np.loadtxt('Dane/dane_8D_diabet_i.txt')
    # y3 = np.loadtxt('Dane/dane_8D_diabet_o.txt')
    x3 = np.loadtxt('Dane/dane_3D_kapitan_i.txt')
    y3 = np.loadtxt('Dane/dane_3D_kapitan_o.txt')

    best_k, best_score, scores = walidacja(x3, y3)
    print(f"Najlepszy parametr k: {best_k} z dokładnością {best_score}")
