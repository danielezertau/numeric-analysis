import math
from scipy.special import factorial
import matplotlib.pyplot as plt
import numpy as np


def iter_root(f, g, x0):
    eps = 1e-8
    max_iter = 1000
    num_iter = 0
    xn = x0
    while abs(f(xn)) >= eps and num_iter < max_iter:
        xn = g(xn)
        num_iter += 1
    return xn


def q1a():
    f = lambda x: math.pow(x, 3) - 5 * math.pow(x, 2)
    g1 = lambda x: x - (f(x) / 25)
    g2 = lambda x: (125 - math.pow(x, 2)) / (15 + x)
    print(iter_root(f, g1, 0))
    print(iter_root(f, g2, -10))


def bisection(f, a, b, n):
    mid = (b + a) / 2
    for i in range(n):
        mid = (b + a) / 2
        if f(a) * f(mid) < 0:
            b = mid
        else:  # f(mid) * f(b) < 0
            a = mid
    return mid


def q1b():
    f = lambda x: math.pow(x, 4) - 2
    print(bisection(f, 0, 2, 21))


def secant_method(f, a, b, eps):
    i = 0
    prev, curr = a, b
    while abs(f(curr)) > eps:
        tmp = curr
        curr = prev - f(prev) * ((curr - prev) / (f(curr) - f(prev)))
        prev = tmp
        i += 1
    return curr, i


def q1c():
    a = 0.5
    bs = [0.500001, 0.6, 0.7]
    eps = 1e-15
    f = lambda x: math.pow(x, 3) + x - 1
    for b in bs:
        r, num_iter = secant_method(f, a, b, eps)
        print(f"b: {b}, result: {r}, num iterations: {num_iter}")


def q1d():
    a = 1 + 0j
    b = 0 + 5j
    f = lambda z: (z ** 3) + z - 1
    comp_root, _ = secant_method(f, a, b, 1e-10)
    return comp_root, np.conj(comp_root)


def cond_num(X, norm):
    return np.linalg.norm(X, ord=norm) * np.linalg.norm(np.linalg.inv(X), ord=norm)


def q2c(eps, delta):
    A = np.array([[1, 1], [0, eps]], dtype=float)
    B = np.array([[1, 0], [1, delta]], dtype=float)
    return float(cond_num(A, norm=2) * float(cond_num(B, norm=2)))


def q3a_factorial(n):
    return (n ** n) / factorial(n)


def q3b():
    for n in range(1, 1002):
        print(f"n: {n}, f(n): {q3a_factorial(n)}")


def q3c_new_factorial(n):
    result = 1
    for k in reversed(range(1, n + 1)):
        result *= (n / k)
    return result


def q3d():
    for n in range(1, 1002):
        print(f"n: {n}, f(n): {q3c_new_factorial(n)}")


def get_a(i, j, n):
    if i == j:
        return n + 1
    else:
        return 1


def init_iter_method(n):
    max_iter = math.pow(n, 2)
    eps = 1e-10
    b = np.ones(n, dtype=np.float64) * (2 * n)
    x_0 = np.zeros_like(b)

    D = np.eye(n, dtype=np.float64) * (n + 1)

    LU = np.ones_like(D) - np.eye(n)
    L = np.tril(LU)
    U = np.triu(LU)

    A = L + D + U

    return max_iter, eps, x_0, b, L, D, U, A


def jacobi(n):
    n_iter = 0
    max_iter, eps, x_k, b, L, D, U, A = init_iter_method(n)
    D_inv = np.eye(n) * 1 / (n + 1)

    while np.linalg.norm((A @ x_k) - b, ord=np.inf) >= eps and n_iter < max_iter:
        x_k = (D_inv @ b) - (D_inv @ (L + U)) @ x_k
        n_iter += 1
    return x_k, n_iter


def gauss_seidel(n):
    n_iter = 0
    max_iter, eps, x_k, b, L, D, U, A = init_iter_method(n)
    DL_inv = np.linalg.inv(D + L)

    while np.linalg.norm((A @ x_k) - b, ord=np.inf) >= eps and n_iter < max_iter:
        x_k = (DL_inv @ b) - (DL_inv @ U) @ x_k
        n_iter += 1
    return x_k, n_iter


def run_iters(func, name):
    ns = np.arange(15, 81)
    results = np.zeros_like(ns, dtype=np.float64)
    num_iters = np.zeros_like(ns)
    for i, n in enumerate(ns):
        actual_sol = np.ones(n)
        estimated_sol, num_iter = func(n)
        results[i] = np.linalg.norm(actual_sol - estimated_sol, ord=np.inf)
        num_iters[i] = num_iter

    plt.title(f"{name.title()} Error")
    plt.xlabel("N")
    plt.ylabel("inf norm ||x - x_k ||")
    plt.plot(ns, results)
    plt.savefig(f"{name}-error.pdf")
    plt.show()

    plt.title(f"{name.title()} Speed")
    plt.xlabel("N")
    plt.ylabel("num iters to convergence")
    plt.plot(ns, num_iters, label=name.title())
    # plt.plot(ns, np.log2(ns) + 13, label="y = log(x) + 13")
    plt.legend()
    plt.savefig(f"{name}-speed-2.pdf")
    plt.show()


def q4a():
    run_iters(jacobi, "jacobi")


def q4b():
    run_iters(gauss_seidel, "gauss-seidel")


if __name__ == '__main__':
    q3b()
