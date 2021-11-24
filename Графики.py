from matrix import *
import matplotlib.pyplot as plt
import csv
from sympy import lambdify, symbols

with open("d://input_inter.csv") as r_file:
    file_reader = csv.reader(r_file, delimiter=";")
    x = []
    y = []
    for row in file_reader:
        x.append(round(type_conversion(row[0].replace(',', '.')), 5))
        y.append(round(type_conversion(row[1].replace(',', '.')), 5))

# Метод Лагранжа
_ = lagranz(x, y)
method_lagranza = _[0]
polinom = _[1]
t = symbols('t')
f_polinom = lambdify(t, polinom)
f_lagranz = []
x_lagranz = []
for i in range(int(min(x) * 10), int(max(x) * 10)):
    x_lagranz.append(i / 10)
    f_lagranz.append(f_polinom(i / 10))


fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True)

ax = fig.axes[0]
ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_lagranz, f_lagranz, 'r', label=f'Интерполированная функция')

ax.set_title("Интерполяция методом Лагранжа ")
ax.legend(loc='best', prop={'size': 8}, frameon = False)


# Интерполяция методом кубического сплайна

splines = BuildSpline(x,y)
f_splines = []
x_splines = []
for i in range(int(min(x) * 10), int(max(x) * 10)):
    x_splines.append(i / 10)
    f_splines.append(Interpolate(splines, i / 10))

ax = fig.axes[1]
ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_splines, f_splines, 'r', label=f'Интерполированная функция')

ax.set_title("Интерполяция методом кубического сплайна ")
ax.legend(loc='best', prop={'size': 8}, frameon = False)


#Аппроксимация линейной функцией

noname = method_lin(x,y)

straight = noname[1]
_ = noname[0]
gamma = round(noname[2],3)
x_lin = []
f_lin = []
for i in range(len(x)):
    x_lin.append(_[i][0])
    f_lin.append(_[i][2])

ax = fig.axes[2]
ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_lin, f_lin, 'r', label=f'Интерполированная функция c G = {gamma}')

ax.set_title("Аппроксимация линейной функцией")
ax.legend(loc='best', prop={'size': 8}, frameon = False)


# Аппрокисмация квардратичной функцией

noname = method_min_square(x,y)
straight = noname[1]
_ = noname[0]
gamma = round(noname[2],3)
x_square = []
f_square = []
for i in range(len(x)):
    x_square.append(_[i][0])
    f_square.append(_[i][2])

ax = fig.axes[3]
ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_square, f_square, 'r', label=f'Интерполированная функция c G = {gamma}')

ax.set_title("Аппрокисмация квардратичной функцией ")
ax.set_ylabel(u'Функция')
ax.legend(loc='best', prop={'size': 8}, frameon = False)


# Аппрокисмация логарифмом

plt.show()
