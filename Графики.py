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

fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True)

ax = fig.axes[0]

ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_lagranz, f_lagranz, 'r', label=f'Интерполированная функция')

ax.set_title("Интерполяция методом Лагранжа ")
ax.legend(loc='best', prop={'size': 8}, frameon = False)

#plt.show()


# Интерполяция методом кубического сплайна

ax = fig.axes[1]
splines = BuildSpline(x,y)
f_splines = []
x_splines = []
for i in range(int(min(x) * 10), int(max(x) * 10)):
    x_splines.append(i / 10)
    f_splines.append(Interpolate(splines, i / 10))


ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_splines, f_splines, 'r', label=f'Интерполированная функция')

ax.set_title("Интерполяция методом кубического сплайна ")
ax.legend(loc='best', prop={'size': 8}, frameon = False)

#plt.show()


#Аппроксимация линейной функцией
ax = fig.axes[2]

noname = method_lin(x,y)

straight = noname[1]
_ = noname[0]
gamma = round(noname[2],3)
x_lin = []
f_lin = []
for i in range(len(x)):
    x_lin.append(_[i][0])
    f_lin.append(_[i][2])


ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_lin, f_lin, 'r', label=f'Аппроксимированная функция c G = {gamma}')

ax.set_title("Аппроксимация линейной функцией")
ax.legend(loc='best', prop={'size': 8}, frameon = False)

#plt.show()

# Аппрокисмация квардратичной функцией
ax = fig.axes[3]
noname = method_min_square(x,y)
straight = noname[1]
_ = noname[0]
gamma = round(noname[2],3)
x_square = []
f_square = []
for i in range(len(x)):
    x_square.append(_[i][0])
    f_square.append(_[i][2])


ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_square, f_square, 'r', label=f'Аппроксимированная функция c G = {gamma}')

ax.set_title("Аппрокисмация квардратичной функцией ")
ax.set_ylabel(u'Функция')
ax.legend(loc='best', prop={'size': 8}, frameon = False)

#plt.show()

# Апроксимация логарифмической функцией   
ax = fig.axes[4]
noname = approximate_log_function(x,y)
gamma = round(noname[1],3)
_ = noname[0]
x_square = []
f_square = []
for i in range(len(x)):
    x_square.append(_[i][0])
    f_square.append(_[i][2])


ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_square, f_square, 'r', label=f'Аппроксимированная функция c G = {gamma}')

ax.set_title("Аппрокисмация логарифмической функцией ")
ax.set_ylabel(u'Функция')
ax.legend(loc='best', prop={'size': 8}, frameon = False)

#plt.show()

#Апроксимация функцией нормального распределения

ax = fig.axes[5]
noname = ap_norm_rasp(x,y)
_ = noname[0]
gamma = round(noname[1],3)
x_square = []
f_square = []
for i in range(len(x)):
    x_square.append(_[i][0])
    f_square.append(_[i][2])

ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_square, f_square, 'r', label=f'Аппроксимированная функция c G = {gamma}')

ax.set_title("Аппрокисмация функцией нормального распределения")
ax.set_ylabel(u'Функция')
ax.legend(loc='best', prop={'size': 8}, frameon = False)

#plt.show()
#Апроксимация экспоненциальной функцией 
ax = fig.axes[6]
noname = approximate_exp_function(x,y)
straight = noname[1]
_ = noname[0]
gamma = round(noname[2],3)
x_square = []
f_square = []
for i in range(len(x)):
    x_square.append(_[i][0])
    f_square.append(_[i][2])


ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_square, f_square, 'r', label=f'Аппрокисимированная функция c G = {gamma}')

ax.set_title("Аппрокисмация экспоненциальной функцией ")
ax.set_ylabel(u'Функция')
ax.legend(loc='best', prop={'size': 8}, frameon = False)

#plt.show()

#Интерполяция стандартной библиотекой
ax = fig.axes[7]
#x_np = interpol_st(x,y)[0]
#y_np = interpol_st(x,y)[1]
x_np = []
y_np = []
for i in (np.linspace(min(x),max(x),1000)):
    x_np.append(i)
    y_np.append(np.interp(i, x, y))

ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_np, y_np, 'r', label=f'Интерполированная функция')
ax.set_title("Интерполяция стандартной библиотекой")
ax.set_ylabel(u'Функция')
ax.legend(loc='best', prop={'size': 8}, frameon = False)


#Аппроксимация стандартной библиотекой
ax = fig.axes[8]
x_np = []
y_np = []
for i in range(int(min(x) * 10), int(max(x) * 10)):
    x_np.append(i / 10)
    y_np.append(np.interp(i / 10, x, y))


#ax.plot(x, y, 'b', label="Исходные точки")
#ax.plot(x_np, y_np, 'r', label=f'Аппроксимированная функция')

ax.set_title("Аппроксимация стандартной библиотекой")
ax.set_ylabel(u'Функция')
ax.legend(loc='best', prop={'size': 8}, frameon = False)


# Интерполяция Ньютоном (прямой ход)
ax = fig.axes[9]
noname = newton_there(x,y)
straight = noname[1]
x_square = []
f_square = []
for i in range(len(noname[1])):
    x_square.append(noname[0][i])
    f_square.append(noname[1][i])

ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x_square, f_square, 'r', label=f'Интерполированная функция')

ax.set_title("Интерполяция Ньтоном (прямой ход) ")
ax.set_ylabel(u'Функция')
ax.legend(loc='best', prop={'size': 8}, frameon = False)


# Интерполяция Ньютоном (обратный ход)
ax = fig.axes[10]
noname = newton_here_the_boss(x,y)
straight = noname[1]
x1_square = []
f1_square = []
for i in range(len(noname[1])):
    x1_square.append(noname[0][i])
    f1_square.append(noname[1][i])


ax.plot(x, y, 'b', label="Исходные точки")
ax.plot(x1_square, f1_square, 'r', label=f'Интерполированная функция')
ax.set_title("Интерполяция Ньтоном (обратный ход) ")
ax.set_ylabel(u'Функция')
ax.legend(loc='best', prop={'size': 8}, frameon = False)

plt.show()
