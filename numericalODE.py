import numpy as np
from scipy.optimize import fsolve


def calculate_initial_s(f, t, h, s0, number_of_initial_values=4):
    return runge_kutta4(f, t[:number_of_initial_values], h, s0)[-1][1:]


def calculate_next_s(f, t, h, s, i, coefficients):
    return h / 24 * np.dot(coefficients, [f(t[i], s[i]), f(t[i], s[i]),
                                          f(t[i - 1], s[i - 1]), f(t[i - 2], s[i - 2]),
                                          f(t[i - 3], s[i - 3])])


def runge_kutta4(f, t, h, s0):
    s = np.zeros(len(t))
    s[0] = s0

    for i in range(len(s) - 1):
        k1 = f(t[i], s[i])
        k2 = f(t[i] + h / 2, s[i] + k1 / 2)
        k3 = f(t[i] + h / 2, s[i] + k2 / 2)
        k4 = f(t[i + 1], s[i] + k3)
        s[i + 1] = s[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, s


def fourth_order_adams_bashforth(f, t, h, s0):
    s = np.zeros(len(t))
    s[0] = s0
    s[1], s[2], s[3] = calculate_initial_s(f, t, h, s0)
    coefficients = [55, -59, 37, -9]
    for i in range(3, len(s) - 1):
        s[i + 1] = s[i] + h / 24 * np.dot(coefficients, [f(t[j], s[j]) for j in range(i, i - 4, -1)])
    return t, s


def fourth_order_adams_moulton(f, t, h, s0):
    s = np.zeros(len(t))
    s[0] = s0
    s[1], s[2], s[3] = calculate_initial_s(f, t, h, s0)
    tmp = s.copy()
    coefficients_bashforth = [55, -59, 37, -9]
    coefficients_moulton = [9, 19, -5, 1]
    for i in range(3, len(s) - 1):
        tmp[i + 1] = s[i] + h / 24 * np.dot(coefficients_bashforth, [f(t[j], s[j]) for j in range(i, i - 4, -1)])
        s[i + 1] = s[i] + h / 24 * np.dot(coefficients_moulton, [f(t[j], tmp[j]) for j in range(i + 1, i - 3, -1)])
    return t, s


def three_stage_diagonally_implicit_runge_kutta_method(f, t, h, s0):
    X = 0.4358665215
    s = np.zeros(len(t))
    s[0] = s0
    s[1], s[2] = calculate_initial_s(f, t, h, s0, 3)
    c = np.array([X, (1 + X) / 2, 1])
    b = np.array([-(3 / 2) * X ** 2 + 4 * X - 1 / 4, (3 / 2) * X ** 2 - 5 * X + 5 / 4, X])
    a = np.array([[X, 0, 0],
                  [(1 - X) / 2, X, 0],
                  [-(3 / 2) * X ** 2 + 4 * X - 1 / 4, (3 / 2) * X ** 2 - 5 * X + 5 / 4, X]])
    for i in range(2, len(t) - 1):
        k_initial_guess = np.array([s[0], s[1], s[2]])

        def equations(k):
            k1, k2, k3 = k
            return [
                k1 - f(t[i] + c[0] * h, s[i] + h * (a[0][0] * k1)),
                k2 - f(t[i] + c[1] * h, s[i] + h * (a[1][0] * k1 + a[1][1] * k2)),
                k3 - f(t[i] + c[2] * h, s[i] + h * (a[2][0] * k1 + a[2][1] * k2 + a[2][2] * k3))
            ]

        result = fsolve(equations, k_initial_guess)
        s[i + 1] = s[i] + h * np.dot(b, result)

    return t, s
