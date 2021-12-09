import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, sqrt, cos, sin, log, exp
from matrix import *




def eiler(func, n=100):
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
            y[k + 1] = y[k] + dx * func[0](x[k], y[k], 0)
            output.append([x[k], y[k]])
        return output
    if answer == '1':
        output = [[0, Y0, Z0]]
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](x[k], z[k], 0)
            z[k + 1] = z[k] + dx * func[1](x[k], z[k], 0)
            output.append([x[k], y[k], z[k]])
        return output


def eiler_Koshi(func, n=100):
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
            y[k + 1] = y[k] + dx * func[0](x[k], y[k], 0)
            y[k + 1] = y[k] + dx / 2 * (func[0](x[k], y[k], 0) + func[0](x[k + 1], y[k + 1], 0))
            output.append([x[k], y[k]])
        return output
    if answer == '1':
        output = [[0, Y0, Z0]]
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](x[k], y[k], z[k])
            y[k + 1] = y[k] + dx / 2 * (func[0](x[k], y[k], 0) + func[0](x[k + 1], y[k + 1], 0))
            z[k + 1] = z[k] + dx * func[1](x[k], z[k], 0)
            z[k + 1] = z[k] + dx / 2 * (func[1](x[k], z[k], 0) + func[1](x[k + 1], z[k + 1], 0))
            output.append([x[k], y[k], z[k]])
        return output


def method_runge_kutta(function):
    y0 = float(input('Начальное условие, y0 = '))
    if answer == '1':
        z0 = float(input('Начальное условие, z0 = '))
        zn = z0
    a = float(input('Начало промежутка: '))
    b = float(input('Конец промежутка: '))
    n = 100
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


def input_function():
    t = symbols('t')
    y = symbols('y')
    z = symbols('z')
    equation = []

    function_y = eval(input('Введите y` [доступные символы t,y,z]: '))
    equation.append(function_y)
    global answer
    answer = input('Хотите ввести еще одно уравнение? [1/0] ').replace(' ', '')
    if answer == '1':
        function_y = eval(input('Введите z` [доступные символы t,y,z]: '))
        equation.append(function_y)
    function = []
    for i in range(len(equation)):
        function.append(lambdify([t, y, z], equation[i]))
    return function

plt.style.use('ggplot')
Input = input_function()
ewq = eiler(Input)
X1 = []
Y1 = []
Z1 = []

for i in range(len(ewq)):
    X1.append(ewq[i][0])
    Y1.append(ewq[i][1])
    if answer == '1':
        Z1.append(ewq[i][2])
plt.plot(X1, Y1, 'm', label='Эйлер')
if answer == '1':
    plt.plot(X1, Z1, 'y',  label='Эйлер')
plt.title("Эйлер")
plt.legend(loc='best', prop={'size': 8}, frameon=False)
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
x2_square = []
f2_square = []
for i in range(len(X1)):
    x1_square.append(_1[i][0])
    f1_square.append(_1[i][2])
    if answer == '1':
        x2_square.append(_2[i][0])
        f2_square.append(_2[i][2])
plt.plot(x1_square, f1_square, 'm', label=f'Интерполированная функция c G = {gamma1}')
if answer == '1':
    plt.plot(x2_square, f2_square, 'm', label=f'Интерполированная функция c G = {gamma2}')
plt.title("МНК Эйлера ")
plt.legend(loc='best', prop={'size': 8}, frameon=False)
plt.show()

noname1 = method_newton(X1, Y1)
if answer == '1':
    noname2 = method_newton(X1, Z1)
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
plt.plot(x1_square, f1_square, 'm', label=f'Апроксимированная функция ')
if answer == '1':
    plt.plot(x2_square, f2_square, 'm', label=f'Апроксимированная функция ')
plt.title("Апроксимация Ньтоном Эйлера ")
plt.legend(loc='best', prop={'size': 8}, frameon=False)
plt.show()

ewq2 = eiler_Koshi(Input)
X2 = []
Y2 = []
Z2 = []

for i in range(len(ewq)):
    X2.append(ewq[i][0])
    Y2.append(ewq[i][1])
    if answer == '1':
        Z2.append(ewq[i][2])
plt.plot(X2, Y2, label='Эйлер- Коши')
if answer == '1':
    plt.plot(X2, Z2, label='Эйлер-Коши')
plt.title("Эйлер-Коши")
plt.legend(loc='best', prop={'size': 8}, frameon=False)
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
x2_square = []
f2_square = []
for i in range(len(X1)):
    x1_square.append(_1[i][0])
    f1_square.append(_1[i][2])
    if answer == '1':
        x2_square.append(_2[i][0])
        f2_square.append(_2[i][2])
plt.plot(x1_square, f1_square, 'm', label=f'Интерполированная функция c G = {gamma1}')
if answer == '1':
    plt.plot(x2_square, f2_square, 'm', label=f'Интерполированная функция c G = {gamma2}')
plt.title("МНК Эйлера-Коши ")
plt.legend(loc='best', prop={'size': 8}, frameon=False)
plt.show()

noname1 = method_newton(X2, Y2)
if answer == '1':
    noname2 = method_newton(X2, Z2)
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
plt.plot(x1_square, f1_square, 'm', label=f'Апроксимированная функция ')
if answer == '1':
    plt.plot(x2_square, f2_square, 'm', label=f'Апроксимированная функция ')
plt.title("Апроксимация Ньтоном Эйлера-Коши ")
plt.legend(loc='best', prop={'size': 8}, frameon=False)
plt.show()

x = []
y = []
z = []

noname = method_runge_kutta(Input)
for i in range(len(noname)):
    x.append(noname[i][1])
    y.append(noname[i][2])
    if answer == '1':
        z.append(noname[i][3])

plt.plot(x, y, label='Рунге-Кутта y')
if answer == '1':
    plt.plot(x, z, label='Рунге-Кутта z')
plt.title("Выходные точки метода Рунге-Кутты")
plt.legend(loc='best', prop={'size': 8}, frameon=False)

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

plt.plot(x_square, y_square, 'm', label=f'Метод наим квадратов y(t) c G = {gamma_y}')
if answer == '1':
    plt.plot(x_square, z_square, 'y', label=f'Метод наим квадратов y(t) c G = {gamma_y}')
plt.title("МНК")
plt.legend(loc='best', prop={'size': 8}, frameon=False)

plt.show()

noname = method_newton(x, y)
if answer == '1':
    nonamez = method_newton(x, z)
x1_square = []
f1_square = []
z_square = []
for i in range(len(noname[1])):
    x1_square.append(noname[0][i])
    f1_square.append(noname[1][i])
    if answer == '1':
        z_square.append(nonamez[1][i])
plt.plot(x1_square, f1_square, 'm', label=f'Метод Ньютона  y')
if answer == '1':
    plt.plot(x1_square, z_square, 'y', label=f'Метод Ньютона  z')
plt.title("Ньютон")
plt.legend(loc='best', prop={'size': 8}, frameon=False)

plt.show()
