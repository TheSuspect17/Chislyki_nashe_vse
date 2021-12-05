import math
def func(x, y):
    return math.cos(x+y)


def rungekutta4(f):
    y0 = float(input('Начальное условие, y0 = '))
    a = float(input('Начало промежутка: '))
    b = float(input('Конец промежутка: '))
    n = 10
    h = (b - a) / n
    output = [[0, 0, y0]]
    counter = 0
    yn = y0
    otrezok = []
    for i in range(n):
        a += h
        otrezok.append(round(a,5))
    for i in otrezok:
        counter += 1
        k1 = f(i, yn)
        k2 = f(i + h / 2, yn + h / 2 * k1)
        k3 = f(i + h / 2, yn + h / 2 * k2)
        k4 = f(i + h, yn + h * k3)
        yn = yn + (h / 6) * (k1 + 2 * k2 + 3 * k3 + k4)
        output.append([counter, i, yn])
    return output
