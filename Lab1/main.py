import numpy as np


def read_data(filename):
    file = open(filename, 'r')
    z = list(map(lambda x: float(x), file.readline().split()))
    line = list(map(lambda x: int(x), file.readline().split()))
    base = []
    end = []
    for i in range(len(line)):
        if line[i] != 0:
            base.append(i)
        else:
            end.append(i)
    base.extend(end)

    a = np.array(list(map(lambda x: float(x), file.readline().split())))
    line = file.readline()
    while line:
        b = np.array(list(map(lambda x: float(x), line.split())))
        a = np.row_stack((a, b))
        line = file.readline()
    return z, a, base


def shift(z, a, base):
    n, m = len(a), len(base)
    a1, z1 = np.copy(a), np.copy(z)
    for i in range(n):
        for j in range(m):
            a1[i, j] = a[i, base[j]]
    for j in range(m):
        z1[j] = z[base[j]]
    return z1, a1


def gauss(a):
    n, m = a.shape
    # traverse
    for i in range(n - 1):
        if a[i, i] == 0:
            for j in range(n):
                if a[j, i] != 0:
                    temp = np.copy(a[i])
                    a[i], a[j] = a[j], temp
                    break
        if a[i, i] == 0:
            print("Error with null base value")
            continue
        a[i] = a[i] / a[i, i]
        for k in range(i + 1, n):
            a[k] = a[k] - (a[i] * a[k, i])
            a[k, i] = 0
    a[n - 1] = a[n - 1] / a[n - 1, n - 1]
    # print(a)
    # reversed traverse
    for i in range(n - 1, -1, -1):
        for k in range(0, i):
            a[k] = a[k] - (a[i] * a[k, i])
            a[k, i] = 0
    for i in range(n):
        for j in range(m):
            if abs(a[i, j]) < 1e-9:
                a[i, j] = 0
    return a


def simplex(a, c):
    m, n = a.shape
    basis_number = np.arange(m)
    d = np.zeros(n)
    z = np.zeros(n)

    for i in range(n - 1):
        for j in range(m):
            z[i] += a[j][i] * c[basis_number[j]]
    z[-1] = 0
    for i in range(n - 1):
        d[i] = z[i] - c[i]
    d[-1] = 0

    check = simplex_step(a, basis_number, d)
    if len(check) == 3:
        a, basis_number, d = check
    else:
        print(check)
        return a, basis_number, d
    result = 0
    for i in range(m):
        result += a[i][-1] * c[basis_number[i]]
    print("Result is:", result)
    return a, basis_number, d


def simplex_step(a, basis_number, d):
    m, n = a.shape
    count = 0
    zero_count = 0
    for i in range(n):
        if abs(d[i]) < 1e-9:
            d[i] = 0
            zero_count += 1
        if d[i] > 0:
            count += 1
    minimum = np.empty(n)
    minimum.fill(1e10)
    if count == 0:
        return a, basis_number, d
    else:
        copy_a = np.copy(a)
        copy_d = np.copy(d)

        max_index = 0
        min_index = 0
        flag = False
        for i in range(n):
            if d[i] > 0:
                min_j_index = 0
                for j in range(m):
                    if a[j][i] > 0 and minimum[i] > a[j][-1] / a[j][i]:
                        minimum[i] = a[j][n - 1] / a[j][i]
                        min_j_index = j
                        flag = True
                if (minimum[max_index] * d[max_index]) < (minimum[i] * d[i]):
                    max_index = i
                    min_index = min_j_index
        if not flag:
            return "Function is not limited at this area"
        for j in range(n):
            copy_a[min_index][j] = a[min_index][j] / a[min_index][max_index]

        for i in range(m):
            if i == min_index:
                continue
            for j in range(n):
                copy_a[i][j] = a[i][j] - (a[min_index][j] / a[min_index][max_index]) * a[i][max_index]
        for i in range(n - 1):
            copy_d[i] = d[i] - (a[min_index][i] / a[min_index][max_index]) * d[max_index]

        copy_d[-1] = d[-1] - (a[min_index][-1] / a[min_index][max_index]) * d[max_index]
        basis_number[min_index] = max_index
        return simplex_step(copy_a, basis_number, copy_d)


def main():
    # c - aim function
    # a - matrix of system of linear equations
    # base - start point
    # test.txt:
    # -6 -1 -4 5
    # 1 0 0 1
    # 3 1 -1 1 4
    # 5 1 1 -1 4
    c, a, base = read_data('./Lab1/test.txt')
    c, a = shift(c, a, base)
    n, m = len(a), len(base)
    a = gauss(a)
    print(a)
    a, basis_number, d = simplex(a, c)
    print(a)
    print(d)
    print(basis_number)


main()
