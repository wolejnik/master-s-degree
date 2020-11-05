from scipy import optimize, cluster
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.cluster.vq import whiten, kmeans, kmeans2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

def main():
    """ Main program """

    # Zadanie 1
    def equations(p):
        x, y, z = p
        return (
            5 / x - 2 / y + 3 / z + 9 / 4,
            2 / x + 3 / y + 1 / z - 5 / 12,
            1 / x + 4 / y - 2 / z - 5 / 3
        )

    x, y, z = fsolve(equations, (-1, 3, 4))
    print(equations((x, y, z)))
    print("\nZadanie 1\n")
    print(f'x:{x}, y:{y}, z:{z}')

    # Zadanie 2

    def min(x):
        return x[0] ** 2 + x[1] ** 2 - x[0] * x[1] + x[0] + x[1]

    def max(x):
        return -(x[0] ** 2 + x[1] ** 2 - x[0] * x[1] + x[0] + x[1])

    cons = (
        {'type': 'ineq', 'fun': lambda x: -x[0]},
        {'type': 'ineq', 'fun': lambda x: -x[1]},
        {'type': 'ineq', 'fun': lambda x: x[1] + x[0] + 3}
    )
    result_min = optimize.minimize(min, (-1, -4), method='SLSQP', constraints=cons)
    result_max = optimize.minimize(max, (-1, 0), method='SLSQP', constraints=cons)
    print("\nZadanie 1\n")
    print(f'Result minimum : {result_min.fun}, result maximum : {result_max.fun}' )

    # Zadanie 3

    def min2(x):
        return -(np.sqrt(4 - x[0] ** 2 - x[1] ** 2))

    def max2(x):
        return np.sqrt(4 - x[0] ** 2 - x[1] ** 2)

    cons = (
        {'type':'ineq', 'fun': lambda x: 4-x[0] ** 2-x[1] ** 2}
    )
    result_min_2 = optimize.minimize(min2, (0, 1), method='SLSQP', constraints=cons)
    result_max_2 = optimize.minimize(max2, (-2, 0), method='SLSQP', constraints=cons)
    print("\nZadanie 3\n")
    print(f'Result minimum 2: {result_min_2.fun}, result maximum 2: {result_max_2.fun}' )

    # Zadanie 4

    def y_1(x):
        return -x + 1

    def y_2(x):
        return x ** 2 - 1

    def y_3(x):
        return np.sin(x)

    x0 = optimize.fsolve(lambda x: y_1(x) - y_3(x), [-0.5])
    x1 = optimize.fsolve(lambda x: y_1(x) - y_3(x), [0.5])
    x2 = optimize.fsolve(lambda x: y_1(x) - y_2(x), [0.5])

    x= np.arange(-3, 3, 0.01)
    y1 = -x + 1
    y3 = np.sin(x)
    plt.plot(x, y_1(x), label=r'$f(x) \quad y < - x + 1$')
    plt.plot(x, y_2(x), label=r'$g(x) \quad y > x^2 - 1$')
    plt.plot(x, y_3(x), label=r'$h(x) \quad y < sin(x)$')
    plt.annotate( r'$x_0$', (x0, y_3(x0)+0.2))
    plt.plot(x0, y_2(x0), 'ko')
    plt.annotate(r'$x_1$', (x1, y_3(x1) + 0.2))
    plt.plot(x1, y_1(x1), 'ko')
    plt.annotate(r'$x_2$', (x2, y_2(x2) + 0.2))
    plt.plot(x2, y_2(x2), 'ko')
    plt.fill_between(x, np.minimum(y_1(x), y_3(x)), y_2(x), where=np.minimum(y_1(x), y_3(x)) > y_2(x), color = 'blue', alpha = 0.5)
    plt.legend()

    # środek cięzkości
    int_step = 0.001
    xx1 = np.arange(x0, x1, int_step)
    A1 = integrate.trapz(y_3(xx1), xx1, int_step)
    xx2 = np.arange(x1, x2, int_step)
    A2 = integrate.trapz(y_1(xx2), xx2, int_step)
    xx3 = np.arange(x0, x2, int_step)
    A3 = integrate.trapz(y_2(xx3), xx3, int_step)
    A = A1 + A2 - A3
    print(f'A: {A}')
    plt.plot(A, -A, 'ko')
    plt.show()

    # Zadanie 5

    # 2D
    M = np.loadtxt('dane_2D.txt')
    wM = whiten(M)
    c, d = kmeans(M, 2)
    c, class1 = kmeans2(M, 2)
    print(c, d, class1)
    plt.scatter(M[:, 0], M[:, 1], marker='o', c=class1)
    plt.plot(c[:, 0], c[:, 1], 'r*')
    plt.show()

    # 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    M = np.loadtxt('dane_3D.txt')
    wM = whiten(M)
    c, d = kmeans(M, 2)
    c, klasa = kmeans2(M, 2)
    print(c, d, klasa)
    ax.scatter(M[:, 0], M[:, 1], M[:, 2], marker='o', c=klasa)
    ax.plot(c[:, 0], c[:, 1], c[:, 2], 'r*')

    return 0

if __name__ == "__main__":
    main()