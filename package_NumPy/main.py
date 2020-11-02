import numpy as np


def addMargins(array, margin):
    """ Zadanie 1
        Add margins into numpy array
        :param array: numpy array
        :param margin: int value

        Returns:
        :return result: numpy array
        """
    result = np.zeros((margin * 2 + array.shape[0], margin * 2 + array.shape[1]))
    result[margin:array.shape[0] + margin, margin:array.shape[1] + margin] = array

    return result

def spiral(n):
    """ Zadanie 2
        A spiral is a matrix (n^2) where the numbers increase go around the edges
        :param n: size of the matrix

        Returns:
        :return result: numpy array
    """
    row1 = 0
    col1 = 0
    row2 = n
    col2 = n
    result = np.ones((n, n))
    num = 1
    while num <= n ** 2:
        for i in range(col1, col2):
            result[row1][i] = num
            num += 1
        if num > n ** 2:
            break
        for i in range(row1 + 1, row2):
            result[i][col2 - 1] = num
            num += 1
        if num > n ** 2:
            break
        for i in range(col2 - 2, col1 - 1, -1):
            result[row2 - 1][i] = num
            num += 1
        if num > n ** 2:
            break
        for i in range(row2 - 2, row1, -1):
            result[i][col1] = num
            num += 1
        row1 += 1
        row2 -= 1
        col1 += 1
        col2 -= 1

    return result


def rangeZeroing(array, a):
    """ Zadanie 3
        Zeroing a value from range <-a, a>
        :param array: numpy array
        :param a: int value

        Returns:
        :return result: numpy array
    """
    # result = np.ones((array.shape[0], array.shape[1]))
    #
    # for x in range(0, array.shape[0]):
    #     for y in range(0, array.shape[1]):
    #         if (-np.abs(a) <= array[x, y] <= np.abs(a)):
    #             result[x, y] = 0

    return np.where(np.logical_and(array <= -a, array <= a), 0, 1)

def deleteRowsColumns(array, a):
    """ Zadanie 4
        Zeroing a value from range <-a, a>
        :param array: numpy array
        :param a: int value

        Returns:
        :return result: numpy array
    """
    return array[np.all(np.logical_and(array >= -a, array <= a), axis=1)]

def uniqueValues(*matrices):
    """ Zadanie 5
        Conversion of the matrix into a vector of unique values
        :param args: matrices

        Returns:
        :return: result copy of the matrices.
    """
    result = []
    flattened = []

    for matrix in matrices:
        result = np.append(result, matrix)

    result, flattened = np.unique(result, return_counts=True)
    result = np.delete(result, np.where(flattened > 1))

    return np.unique(result)


def main():
    """ Main program """

    # Zadanie 1
    matrix = np.ones((4, 5))
    # print(matrix)
    matrixAddedMargins = addMargins(matrix, 2)
    print(f'Result 1:\n {matrixAddedMargins}\n')

    # # Zadanie 2
    result2 = spiral(5)
    print(f'Result 2:\n {result2}\n')

    # Zadanie 3
    A = np.random.randint(low=-10, high=10, size=(5, 5))
    print(f'A:\n {A}\n')
    result3 = rangeZeroing(A, 4)
    print(f'Result 3:\n {result3}\n')

    # Zadanie 4
    B = np.random.randint(low=-8, high=8, size=(8, 8))
    print(f'B:\n {B}\n')
    result4 = deleteRowsColumns(B, 7)
    print(f'Result 4:\n {result4}\n')

    # Zadanie 5
    E = np.random.randint(low=0, high=50, size=(8, 8))
    F = np.random.randint(low=0, high=50, size=(4, 4))
    G = np.random.randint(low=0, high=50, size=(2, 3))
    H = np.random.randint(low=0, high=50, size=(3, 5))
    result5 = uniqueValues(E, F, G, H).astype("int")
    print(f'Result 5:\n {result5}\n')

    return 0


if __name__ == "__main__":
    main()