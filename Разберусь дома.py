import math
import numpy as np
import matplotlib.pyplot as plt
import math
from matrix import method_min_square, newton_here_the_boss
from sympy import symbols, lambdify, sqrt, cos, sin, log, exp

answer = '0'
def diff_left_side(x,y):
    h = x[1] - x[0]
    dy = []
    for i in range(1,len(x)):
        dy.append((y[i] - y[i-1])/h)
    return dy

def func(x, y):
    return math.sqrt(x) + y
def input_function():
    t = symbols('t')
    y = symbols('y')
    z = symbols('z')
    equation = []

    qwe = eval(input('Введите y` [доступные символы t,y,z]: '))
    equation.append(qwe)
    global answer
    answer = input('Хотите ввести еще одно уравнение? [1/0] ')
    if answer == '1':
        qwe = eval(input('Введите z` [доступные символы t,y,z]: '))
        equation.append(qwe)
    function = []
    for i in range(len(equation)):
        function.append(lambdify([t, y, z], equation[i]))
    return function

def rungekutta4(function):
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
            k1 = function[0](i, yn,1) * h
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

x = []
y = []
z = []

function = input_function()

noname = rungekutta4(function)
for i in range(len(noname)):
    x.append(noname[i][1])
    y.append(noname[i][2])
    if answer == '1':
        z.append(noname[i][3])




plt.plot(x, y, label='Выходные точки Рунге-Кутты функции y(t)')
if answer == '1':
    plt.plot(x, z, label='Выходные точки Рунге-Кутты функции z(t)')
plt.title("Выходные точки метода Рунге-Кутты")

plt.legend(loc='best', prop={'size': 8}, frameon=False)

plt.show()

noname = method_min_square(x, y)
_ = noname[0]
if answer == "1":
    nonamez = method_min_square(x,z)
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


plt.plot(x_square, y_square, 'r', label=f'МНК y(t) c G = {gamma_y}')
if answer == '1':
    plt.plot(x_square, z_square, 'b', label=f'МНК z(t) c G = {gamma_z}')
plt.title("МНК")
plt.legend(loc='best', prop={'size': 8}, frameon=False)

plt.show()

noname = newton_here_the_boss(x, y)
if answer == '1':
    nonamez = newton_here_the_boss(x,z)
x1_square = []
f1_square = []
z_square = []
for i in range(len(noname[1])):
    x1_square.append(noname[0][i])
    f1_square.append(noname[1][i])
    if answer == '1':
        z_square.append(nonamez[1][i])
plt.plot(x1_square, f1_square, 'r', label=f'Ньютон y')
if answer == '1':
    plt.plot(x1_square, z_square, 'b', label=f'Ньютон z')
plt.title("Ньютон")
plt.legend(loc='best', prop={'size': 8}, frameon=False)


plt.show()


delta = []
dy_left_side = diff_left_side(x,y)
for i in range(len(dy_left_side)):
    delta.append(abs(dy_left_side[i] - function[0](x[i],y[i],1)))
plt.plot(x[:-1],delta,'b', label = 'Погрешность')
plt.legend(loc='best', prop={'size': 8}, frameon=False)
plt.show()
