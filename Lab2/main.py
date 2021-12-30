import numpy as np

def read_data(filename):
    file = open(filename, 'r')
    a = np.array(list(map(lambda x: float(x), file.readline().split())))
    line = file.readline()
    while line:
        b = np.array(list(map(lambda x: float(x), line.split())))
        a = np.row_stack((a, b))
        line = file.readline()
    return a


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
    a, basis_number, d = find_base(a, basis_number, d)
    for i in range(m):
        for j in range(n):
            if abs(a[i][j]) < 1e-10:
                a[i][j] = 0
            elif abs(a[i][j] - 1) < 1e-10:
                a[i][j] = 1
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
    return a, basis_number, c


def simplex_step(a, basis_number, d):
    m, n = a.shape
    count = 0
    zero_count = 0
    for i in range(n - 1):
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
        for i in range(n - 1):
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

        # for j in range(n):
        #     copy_a[min_index][j] = a[min_index][j] / a[min_index][max_index]
        #
        # for i in range(m):
        #     if i == min_index:
        #         continue
        #     for j in range(n):
        #         copy_a[i][j] = a[i][j] - (a[min_index][j] / a[min_index][max_index]) * a[i][max_index]
        # for i in range(n - 1):
        #     copy_d[i] = d[i] - (a[min_index][i] / a[min_index][max_index]) * d[max_index]
        #
        # copy_d[-1] = d[-1] - (a[min_index][-1] / a[min_index][max_index]) * d[max_index]
        # basis_number[min_index] = max_index
        a, basis_number, d = calc_a(a, basis_number, d, min_index, max_index)
        return simplex_step(a, basis_number, d)


def find_base(a, basis_number, d):
    m, n = a.shape
    min_index = 0
    for i in range(1, m):
        if a[i][-1] < 0 and a[min_index][-1] > a[i][-1]:
            min_index = i
    if a[min_index][-1] >= 0:
        return a, basis_number, d
    for j in range(n - 1):
        if a[min_index][j] < 0:
            a, basis_number, d = calc_a(a, basis_number, d, min_index, j)
            return find_base(a, basis_number, d)
    print("Can't choose the base")
    return a, basis_number, d


def calc_a(a, basis_number, d, ch_row, ch_column):
    m, n = a.shape
    copy_a = np.copy(a)
    copy_d = np.copy(d)
    for j in range(n):
        copy_a[ch_row][j] = a[ch_row][j] / a[ch_row][ch_column]

    for i in range(m):
        if i == ch_row:
            continue
        for j in range(n):
            copy_a[i][j] = a[i][j] - (a[ch_row][j] / a[ch_row][ch_column]) * a[i][ch_column]
    for i in range(n - 1):
        copy_d[i] = d[i] - (a[ch_row][i] / a[ch_row][ch_column]) * d[ch_column]

    copy_d[-1] = d[-1] - (a[ch_row][-1] / a[ch_row][ch_column]) * d[ch_column]
    basis_number[ch_row] = ch_column
    return copy_a, basis_number, copy_d



def saddle_point(a):
    n, m = a.shape
    max_min = -1e10
    for i in range(n):
        # min_val = a[i][0]
        min_val = min(a[i])
        max_min = max(max_min, min_val)

    min_max = 1e10
    for j in range(m):
        # max_val = a[0][j]
        column = a[:, j]
        max_val = max(a[:, j])
        # for i in range(n):
        #     max_val = max(max_val, a[i][j])
        min_max = min(min_max, max_val)

    if min_max != max_min:
        return False, -1
    else:
        return True, min_max


def dominate_check(b):
    flag = True
    a = np.copy(b)
    while flag:
        flag = False
        for i in range(a.shape[0]):
            for k in range(i + 1, a.shape[0]):
                cnt = 0
                for j in range(a.shape[1]):
                    if a[i][j] >= a[k][j] and cnt >= 0:
                        cnt += 1
                    elif a[i][j] <= a[k][j] and cnt <= 0:
                        cnt -= 1
                    else:
                        break
                new_a = a
                if cnt == a.shape[1]:
                    new_a = np.delete(a, k, axis=0)
                elif cnt == -a.shape[1]:
                    new_a = np.delete(a, i, axis=0)
                if abs(cnt) == a.shape[1]:
                    flag = True
                    a = new_a
                    break

        for j in range(a.shape[1]):
            for k in range(j + 1, a.shape[1]):
                cnt = 0
                for i in range(a.shape[0]):
                    if a[i][j] >= a[i][k] and cnt >= 0:
                        cnt += 1
                    elif a[i][j] <= a[i][k] and cnt <= 0:
                        cnt -= 1
                    else:
                        break
                new_a = a
                if cnt == a.shape[0]:
                    new_a = np.delete(a, j, axis=1)
                elif cnt == -a.shape[0]:
                    new_a = np.delete(a, k, axis=1)
                if abs(cnt) == a.shape[0]:
                    flag = True
                    a = new_a
                    break
    return a


def get_strategies(a, player):

    n, m = a.shape
    c = np.zeros((m + n))
    if player == 1:
        c[0: m] = -1
    else:
        c[0: m] = 1
    base = list(range(m, m + n))
    base.extend(list(range(m)))
    new_a = np.zeros((n, m + n))
    if player == 1:
        for i in range(n):
            new_a[i][0: m] = a[i]
            new_a[i][m + i] = 1
    else:
        for i in range(n):
            new_a[i][0: m] = -a[i]
            new_a[i][m + i] = 1
    a = new_a
    c, a = shift(c, a, base)
    k = np.ones((n, 1))
    if player != 1:
        k *= -1
    a = np.hstack((a, k))
    # a = gauss(a)
    # print(a)
    a, basis_number, c = simplex(a, c)
    result = 0
    for i in range(n):
        result += a[i][-1] * c[basis_number[i]]

    if result != 0:
        if player == 1:
            result = -1 / result
        else:
            result = 1 / result
        print("Game value is:", result)
        for i in range(n):
            if c[basis_number[i]] != 0:
                print("x" + str(basis_number[i] - n + 1), " is ", str(a[i][-1] * result))
    else:
        print("There is no specific game value")



def main():
    # c - aim function
    # base - start point
    # a - matrix of system of linear equations

    matr = read_data('./test.txt')

    # a = gauss(a)
    # print(a)
    check, value = saddle_point(matr)
    if check:
        print("Saddle point value:", value)
        return
    matr = dominate_check(matr)
    print("Player 1:")
    get_strategies(matr, 1)
    # print(a)
    # print(d)
    # print(basis_number)
    print("\nPlayer 2:")
    get_strategies(np.transpose(matr), 2)



main()
