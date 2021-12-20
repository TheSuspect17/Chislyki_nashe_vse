import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, sqrt, cos, sin, log, exp
from matrix import *
from scipy.integrate import odeint
from math import sqrt, cos, sin, log, exp

n = 170

answer = '0'
def input_function():
    global qwe654
    qwe654 = input('Введите y` [доступные символы t,y,z]: ')
    global answer
    answer = input('Хотите ввести еще одно уравнение? [1/0] ')
    function = []

    def funct_1(y=0, t=0, z=0, qwe=qwe654):
        dydt = eval(qwe654)
        return dydt

    function.append(funct_1)

    if answer == '1':
        global qwe456
        qwe456 = input('Введите z` [доступные символы t,y,z]: ')

        def funct_2(t=0, y=0, z=0, qwe=qwe456):
            dydt = eval(qwe456)
            return dydt

        function.append(funct_2)
    return function


def euler(func, n=100):
    """Решение ОДУ u'=f(y,x), начальное условие y(0) = U0 , c n шагами, пока  x = b - конец отрезка интегрирования."""
    Y0 = float(input('Начальное условие, y0 = '))
    a = float(input('Начало промежутка: '))
    b = float(input('Конец промежутка: '))
    if answer == '1':
        Z0 = float(input('Начальное условие, z0 = '))
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    y[0] = Y0
    if answer == '1':
        z[0] = Z0
    x[0] = a
    output = [[0, Y0]]
    dx = b / float(n)
    if answer == '0':
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](t=x[k], y=y[k], z=0)
            output.append([x[k], y[k]])
        return output
    if answer == '1':
        output = [[0, Y0, Z0]]
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](t=x[k], z=z[k], y=0)
            z[k + 1] = z[k] + dx * func[1](t=x[k], z=z[k], y=0)
            output.append([x[k], y[k], z[k]])
        return output


def euler_Koshi(func, n=100):
    """Решение ОДУ u'=f(y,x), начальное условие y(0) = U0 , c n шагами, пока  x = b - конец отрезка интегрирования."""
    Y0 = float(input('Начальное условие, y0 = '))
    a = float(input('Начало промежутка: '))
    b = float(input('Конец промежутка: '))
    if answer == '1':
        Z0 = float(input('Начальное условие, z0 = '))
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    y[0] = Y0
    if answer == '1':
        z[0] = Z0
    x[0] = a
    output = [[0, Y0]]
    dx = b / n
    if answer == '0':
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](t=x[k], y=y[k], z=0)
            y[k + 1] = y[k] + dx / 2 * (func[0](t=x[k], y=y[k], z=0) + func[0](t=x[k + 1], y=y[k + 1], z=0))
            output.append([x[k], y[k]])
        return output
    if answer == '1':
        output = [[0, Y0, Z0]]
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](t=x[k], y=y[k], z=z[k])
            y[k + 1] = y[k] + dx / 2 * (func[0](t=x[k], y=y[k], z=0) + func[0](t=x[k + 1], y=y[k + 1], z=0))
            z[k + 1] = z[k] + dx * func[1](t=x[k], z=z[k], y=0)
            z[k + 1] = z[k] + dx / 2 * (func[1](t=x[k], z=z[k], y=0) + func[1](t=x[k + 1], z=z[k + 1], y=0))
            output.append([x[k], y[k], z[k]])
        return output


def rungekutta4(function, n=100):
    y0 = float(input('Начальное условие, y0 = '))
    if answer == '1':
        z0 = float(input('Начальное условие, z0 = '))
        zn = z0
    a = float(input('Начало промежутка: '))
    b = float(input('Конец промежутка: '))
    h = (b - a) / n
    output = [[0, 0, y0]]
    counter = 0
    yn = y0
    otrezok = []
    for i in range(n):
        a += h
        otrezok.append(round(a, 5))
    if answer == '0':
        for i in otrezok:
            counter += 1
            k1 = function[0](i, yn, 1) * h
            k2 = function[0](i + h / 2, yn + k1 / 2, 1) * h
            k3 = function[0](i + h / 2, yn + k2 / 2, 1) * h
            k4 = function[0](i + h, yn + k3, 1) * h
            yn = yn + (1 / 6) * (k1 + 2 * k2 + 3 * k3 + k4)
            output.append([counter, i, yn])
        return output
    elif answer == '1':
        output = [[0, 0, y0, z0]]
        for i in otrezok:
            counter += 1
            k1 = function[0](i, yn, zn) * h
            m1 = function[1](i, yn, zn) * h

            k2 = function[0](i + h / 2, yn + k1 / 2, zn + m1 / 2) * h
            m2 = function[1](i + h / 2, yn + k1 / 2, zn + m1 / 2) * h

            k3 = function[0](i + h / 2, yn + k2 / 2, zn + m2 / 2) * h
            m3 = function[1](i + h / 2, yn + k2 / 2, zn + m2 / 2) * h

            k4 = function[0](i + h, yn + k3, zn + m3) * h
            m4 = function[1](i + h, yn + k3, zn + m3) * h

            yn = yn + (1 / 6) * (k1 + 2 * k2 + 3 * k3 + k4)
            zn = zn + (1 / 6) * (m1 + 2 * m2 + 3 * m3 + m4)
            output.append([counter, i, yn, zn])
        return output


def diff_left_side(x, y):
    h = x[1] - x[0]
    dy = []
    for i in range(1, len(x)):
        dy.append((y[i] - y[i - 1]) / h)
    return dy


plt.style.use('ggplot')

Input = input_function()
qwe = euler(Input, n)
X1 = []
Y1 = []
Z1 = []

for i in range(len(qwe)):
    X1.append(qwe[i][0])
    Y1.append(qwe[i][1])
    if answer == '1':
        Z1.append(qwe[i][2])

if answer != '1':
    plt.plot(X1, Y1, label='Эйлер')
    plt.title("Эйлер")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()

if answer == '1':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, Y1, Z1)
    plt.show()

noname1 = method_min_square(X1, Y1)
if answer == '1':
    noname2 = method_min_square(X1, Z1)
_1 = noname1[0]
if answer == '1':
    _2 = noname2[0]
gamma1 = round(noname1[2], 3)
if answer == '1':
    gamma2 = round(noname2[2], 3)
x1_square = []
f1_square = []
f2_square = []
for i in range(len(X1)):
    x1_square.append(_1[i][0])
    f1_square.append(_1[i][2])
    if answer == '1':
        f2_square.append(_2[i][2])
if answer != '1':
    plt.plot(x1_square, f1_square, 'r', label=f'Интерполированная функция c G = {gamma1}')
    plt.title("МНК Эйлера ")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()
if answer == '1':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1_square, f1_square, f2_square)
    plt.show()

plt.show()

noname1 = newton_here_the_boss(X1, Y1)
if answer == '1':
    noname2 = newton_here_the_boss(X1, Z1)
x1_square = []
f1_square = []
f2_square = []
for i in range(len(noname1[1])):
    x1_square.append(noname1[0][i])
    f1_square.append(noname1[1][i])
    if answer == '1':
        f2_square.append(noname2[1][i])
if answer != '1':
    plt.plot(x1_square, f1_square, 'r', label=f'Апроксимированная функция ')
    plt.title("Апроксимация Ньтоном Эйлера ")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()
if answer == '1':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1_square, f1_square, f2_square)
    plt.show()

qwe2 = euler_Koshi(Input, n)
X2 = []
Y2 = []
Z2 = []

for i in range(len(qwe)):
    X2.append(qwe[i][0])
    Y2.append(qwe[i][1])
    if answer == '1':
        Z2.append(qwe[i][2])
if answer != '1':
    plt.plot(X2, Y2, label='Эйлер- Коши')
    plt.title("Эйлер-Коши")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()

if answer == '1':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X2, Y2, Z2)
    plt.show()

noname1 = method_min_square(X2, Y2)
if answer == '1':
    noname2 = method_min_square(X2, Z2)
_1 = noname1[0]
if answer == '1':
    _2 = noname2[0]
gamma1 = round(noname1[2], 3)
if answer == '1':
    gamma2 = round(noname2[2], 3)
x1_square = []
f1_square = []
f2_square = []
for i in range(len(X1)):
    x1_square.append(_1[i][0])
    f1_square.append(_1[i][2])
    if answer == '1':
        f2_square.append(_2[i][2])
if answer != '1':
    plt.plot(x1_square, f1_square, 'r', label=f'Интерполированная функция c G = {gamma1}')
    plt.title("МНК Эйлера-Коши ")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()
if answer == '1':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1_square, f1_square, f2_square)
    plt.show()


noname1 = newton_here_the_boss(X2, Y2)
if answer == '1':
    noname2 = newton_here_the_boss(X2, Z2)
x1_square = []
f1_square = []
x2_square = []
f2_square = []
for i in range(len(noname1[1])):
    x1_square.append(noname1[0][i])
    f1_square.append(noname1[1][i])
    if answer == '1':
        x2_square.append(noname2[0][i])
        f2_square.append(noname2[1][i])
if answer !='1':
    plt.plot(x1_square, f1_square, 'r', label=f'Апроксимированная функция ')
    plt.title("Апроксимация Ньтоном Эйлера-Коши ")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()
if answer == '1':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1_square, f1_square, f2_square)
    plt.show()


x = []
y = []
z = []

noname = rungekutta4(Input)
for i in range(len(noname)):
    x.append(noname[i][1])
    y.append(noname[i][2])
    if answer == '1':
        z.append(noname[i][3])
if answer != '1':
    plt.plot(x, y, label='Выходные точки Рунге-Кутты функции y(t)')
    plt.title("Выходные точки метода Рунге-Кутты")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()

if answer == '1':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()


noname = method_min_square(x, y)
_ = noname[0]
if answer == "1":
    nonamez = method_min_square(x, z)
    _z = nonamez[0]
    gamma_z = round(nonamez[2], 3)
gamma_y = round(noname[2], 3)
x_square = []
y_square = []
z_square = []
for i in range(len(x)):
    x_square.append(_[i][0])
    y_square.append(_[i][2])
    if answer == '1':
        z_square.append(_z[i][2])

if answer != '1':
    plt.plot(x_square, y_square, 'r', label=f'МНК y(t) c G = {gamma_y}')
    plt.title("МНК")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()
if answer == '1':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_square, y_square, z_square)
    plt.show()


noname = newton_here_the_boss(x, y)
if answer == '1':
    nonamez = newton_here_the_boss(x, z)
x1_square = []
f1_square = []
z_square = []
for i in range(len(noname[1])):
    x1_square.append(noname[0][i])
    f1_square.append(noname[1][i])
    if answer == '1':
        z_square.append(nonamez[1][i])
if answer != '1':
    plt.plot(x1_square, f1_square, 'r', label=f'Ньютон y')
    plt.title("Ньютон")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()
if answer == '1':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1_square, f1_square, z_square)
    plt.show()


delta = []
dy_left_side_y = diff_left_side(x, y)
if answer != '1':
    for i in range(len(dy_left_side_y)):
        delta.append(dy_left_side_y[i] - Input[0](x[i], y[i], 1))
    summa_y = sum(delta)
if answer == '1':
    delta_z = []
    dy_left_side_z = diff_left_side(x, z)
    for i in range(len(dy_left_side_y)):
        delta.append(dy_left_side_y[i] - Input[0](x[i], y[i], z[i]))
        delta_z.append(dy_left_side_z[i] - Input[1](x[i], y[i], z[i]))
    summa_y = sum(delta)
    summa_z = sum(delta_z)
    plt.plot(x[:-1], delta_z, 'r', label=f'Погрешность z sum = {summa_z}')
    plt.legend(loc='best', prop={'size': 8}, frameon=False)


plt.plot(x[:-1], delta, 'b', label=f'Погрешность y sum = {summa_y}')
plt.legend(loc='best', prop={'size': 8}, frameon=False)
plt.show()

y0 = float(input('Введите начальное условие y0 = '))
a = int(input('Начало промежутка '))
b = int(input('Конец промежутка '))
x = np.linspace(a, b, n)

y = odeint(Input[0], y0, x)
if answer != '1':
    plt.plot(x, y, 'm', label='Стандартная функция y`')
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()

if answer == '1':
    z0 = float(input('Введите начальное условие z0 = '))
    z = odeint(Input[0], z0, x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()

