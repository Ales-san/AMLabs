import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    file = open(filename, 'r')
    a = np.array(list(map(lambda x: float(x), file.readline().split())))
    line = file.readline()
    while line:
        b = np.array(list(map(lambda x: float(x), line.split())))
        a = np.row_stack((a, b))
        line = file.readline()
    return a


# π * P = π
# P_transposed * π_transposed = π_transposed
# P_t * π_t - π_t = 0
# (P_t - I) * π_t = 0
def get_analytical_solution(P):
    size = P.shape[0]

    # Calculate left part
    left_part = (P.transpose() - np.eye(size))

    # Update left part with condition of normalization
    # π_1 + ... + π_n = 1
    left_part[-1] = np.ones(size)

    # Build right part
    right_part = np.zeros(size)
    right_part[-1] = 1

    # Get result
    result = np.linalg.solve(left_part, right_part)
    return result


def get_numeric_solution(P, start, steps, EPS):
    cur_eps = 2 * EPS
    cur_p = start
    cur_step = 0
    while cur_eps > EPS and cur_step < steps:
        prev_p = np.copy(cur_p)
        cur_p = cur_p.dot(P)
        cur_eps = calculate_eps(prev_p, cur_p)
        cur_step += 1
    return cur_p


def get_numeric_graph_solution(P, start, steps, EPS, standard):
    cur_eps = 2 * EPS
    cur_p = start
    graph_data = list()
    cur_step = 0
    while cur_eps > EPS and cur_step < steps:
        prev_p = np.copy(cur_p)
        cur_p = cur_p.dot(P)
        cur_eps = calculate_eps(prev_p, cur_p)
        graph_data.append(calculate_standard_deviation(cur_p, standard))
        cur_step += 1
    return cur_p, graph_data


def calculate_eps(solution_1, solution_2):
    return max(abs(solution_2 - solution_1))


def are_equal(solution_1, solution_2, EPS):
    if calculate_eps(solution_1, solution_2) <= EPS:
        return True
    return False


def calculate_standard_deviation(data, standard):
    return np.sqrt(np.sum(np.square(data - standard)) / data.shape[0])


def plot_eps(graph_1, graph_2):
    fig, ax = plt.subplots()
    ax.set_xlabel('iterations')
    ax.set_ylabel('deviation')
    ax.plot(graph_1, label='first')
    ax.plot(graph_2, label='second')
    ax.scatter(range(len(graph_1)), graph_1)
    ax.scatter(range(len(graph_2)), graph_2)
    plt.legend()
    plt.show()


def main():
    P = read_data('./test.txt')
    EPS = 1e-6
    steps = 1e4

    start1 = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    start2 = np.array([0.05, 0.2, 0.1, 0.2, 0.05, 0.15, 0.1, 0.15])

    standard = get_analytical_solution(P)
    numerical_solution_1 = get_numeric_graph_solution(P, start1, steps, EPS, standard)
    numerical_solution_2 = get_numeric_graph_solution(P, start2, steps, EPS, standard)

    print("Analytical solution is:")
    print(standard)
    print("Numerical solution 1 is:")
    print(numerical_solution_1[0])
    print("Numerical solution 2 is:")
    print(numerical_solution_2[0])
    if are_equal(standard, numerical_solution_1[0], EPS):
        print("\nSolutions (analytical and first numerical) are equal with bias of", EPS)
    else:
        print("\nSolutions (analytical and first numerical) differ more than", EPS)
    if are_equal(standard, numerical_solution_2[0], EPS):
        print("Solutions (analytical and second numerical) are equal with bias of", EPS)
    else:
        print("Solutions (analytical and second numerical) differ more than", EPS)

    plot_eps(numerical_solution_1[1], numerical_solution_2[1])


main()
