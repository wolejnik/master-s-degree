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



    return 0

if __name__ == "__main__":
    main()