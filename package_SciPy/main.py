from scipy import optimize, cluster
from scipy.optimize import fsolve


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
    print(f' Result minimum : {result_min.fun}, result maximum : {result_max.fun}' )



    return 0

if __name__ == "__main__":
    main()