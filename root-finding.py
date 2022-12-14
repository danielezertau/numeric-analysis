import math


def is_zero(x, eps):
    return abs(x) < eps


def sanity(f, a, b, eps):
    if is_zero(f(a), eps):
        return a
    elif is_zero(f(b), eps):
        return b
    elif f(a) * f(b) > 0:
        raise Exception(f"No root in [{a}, {b}]")


def bisection(f, a, b, eps):
    res = sanity(f, a, b, eps)
    if res is not None:
        return res
    mid = (b + a) / 2
    while abs(f(mid)) > eps:
        mid = (b + a) / 2
        if f(a) * f(mid) < 0:
            b = mid
        else:  # f(mid) * f(b) < 0
            a = mid
    return mid


def newton_raphson(a, b, eps):
    f = lambda x: 230 * math.pow(x, 4) + 18 * math.pow(x, 3) + 9 * math.pow(x, 2) - 221 * x - 9
    f_prime = lambda x: 4 * 230 * math.pow(x, 3) + 3 * 18 * math.pow(x, 2) + 2 * 9 * x - 221
    res = sanity(f, a, b, eps)
    if res is not None:
        return res
    x_n = (b + a) / 2
    while not is_zero(f(x_n), eps):
        x_n = x_n - (f(x_n) / f_prime(x_n))
    return x_n


def false_position(f, a, b, eps):
    res = sanity(f, a, b, eps)
    if res is not None:
        return res
    calc_xn = lambda a_n, b_n: (a_n * f(b_n) - b_n * f(a_n)) / (f(b_n) - f(a_n))
    x_n = calc_xn(a, b)

    while not is_zero(f(x_n), eps):
        if f(a) * f(x_n) < 0:
            b = x_n
        else:  # f(x_n) * f(b) < 0
            a = x_n
        x_n = calc_xn(a, b)
    return x_n


def secant_method(f, a, b, eps):
    prev, curr = a, b
    while not is_zero(f(curr), eps):
        tmp = curr
        curr = prev - f(prev) * ((curr - prev) / (f(curr) - f(prev)))
        prev = tmp
    return curr


if __name__ == '__main__':
    epsilon = 1e-10
    func = lambda x: 230 * math.pow(x, 4) + 18 * math.pow(x, 3) + 9 * math.pow(x, 2) - 221 * x - 9
    # Newton-Raphson
    print("Newton-Raphson:")
    print(newton_raphson(0, 100, epsilon))
    print(newton_raphson(-1, 0, epsilon))
    # Secant Method
    print("Secant Method:")
    print(secant_method(func, 1, 2, epsilon))
    print(secant_method(func, -1, 0, epsilon))

    # False Position
    print("False Position:")
    print(false_position(func, 0, 1, epsilon))
    print(false_position(func, -1, 0, epsilon))
