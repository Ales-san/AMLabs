from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
import numpy as np

x_p = []
kl_p = []
kdt_p = []


def nstatedes(s, p, l, u):
    kload = p[1] + p[2] + p[3]
    kdtime = p[0]

    x_p.append(s)
    kl_p.append(kload)
    kdt_p.append(kdtime)

    return [-l * p[0] + u * p[1],
            l * (p[0] - p[1]) + u * (p[2] - p[1]),
            l * (p[1] - p[2]) + u * (p[3] - p[2]),
            l * p[2] - u * p[3]]


def main():
    l = 4.46
    u = 2.7
    # result = solve_ivp(nstatedes, (0, 3), [1, 0, 0, 0], args=(l, u))
    t = np.arange(0, 3, 0.001)
    result = odeint(nstatedes, [1., 0., 0., 0.], t, (l, u), tfirst=True)
    print(result)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    for i in range(4):
        ax.plot(t, result[:, i], label="p" + str(i))
    ax.set_title("probability")
    ax.grid()
    ax.legend()

    print(kl_p[-10:])
    print(kdt_p[-10:])
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_p, kl_p, label="load")
    ax.plot(x_p, kdt_p, label="downtime")
    ax.set_title("coefficients")
    ax.grid()
    ax.legend()
    plt.show()


main()