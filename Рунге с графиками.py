import matplotlib.pyplot as plt
import math
from matrix import method_min_square, newton_here_the_boss
def func(x, y):
    return math.exp(-x) - y



def rungekutta4(f):
    y0 = float(input('Начальное условие, y0 = '))
    a = float(input('Начало промежутка: '))
    b = float(input('Конец промежутка: '))
    global n
    n = 100
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


x = []
y = []

noname = rungekutta4(func)
for i in range(n):
    x.append(noname[i][1])
    y.append(noname[i][2])



fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)

ax = fig.axes[0]

ax.plot(x,y, label = 'Выходные точки Рунге-Кутты')
ax.set_title("Выходные точки метода Рунге-Кутты")
ax.legend(loc='best', prop={'size': 8}, frameon = False)

ax = fig.axes[1]
noname = method_min_square(x,y)
_ = noname[0]
gamma = round(noname[2],3)
x_square = []
f_square = []
for i in range(len(x)):
    x_square.append(_[i][0])
    f_square.append(_[i][2])


ax.plot(x_square, f_square, 'r', label=f'МНК c G = {gamma}')
ax.set_title("МНК")
ax.legend(loc='best', prop={'size': 8}, frameon = False)


ax = fig.axes[2]
noname = newton_here_the_boss(x,y)
straight = noname[1]
x1_square = []
f1_square = []
for i in range(len(noname[1])):
    x1_square.append(noname[0][i])
    f1_square.append(noname[1][i])


ax.plot(x1_square, f1_square, 'r', label=f'Ньютон')
ax.set_title("Ньютон")
ax.legend(loc='best', prop={'size': 8}, frameon = False)

plt.show()



