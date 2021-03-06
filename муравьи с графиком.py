from random import uniform, randint
from fractions import Fraction
import numpy as np
from scipy.optimize import leastsq
from sympy import symbols, expand, lambdify
import random
import math
from scipy.optimize import leastsq, curve_fit
from sklearn.metrics import mean_squared_error
import random as rd
import itertools
import networkx as nx
import numpy.random as rnd
import matplotlib.pyplot as plt
import time

maxsize = float('inf')


def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        try:
            complex(str)
            return True
        except ValueError:
            return False
    except TypeError:
        try:
            complex(str)
            return True
        except TypeError:
            return False


# .is_number ( str = string )
# return: TRUE если str может быть преобразован к типу complex или float

def type_conversion(str):  # преобразование типов
    try:
        float(str)
        return float(str)
    except ValueError:
        complex(str)
        return complex(str)


# .type_conversion(str = string)
# Возвращает float(str), при ошибке типа ValueError complex(str)

def matrixTranspose(anArray):
    transposed = [None] * len(anArray[0])
    for t in range(len(anArray)):
        transposed[t] = [None] * len(anArray)
        for tt in range(len(anArray[t])):
            transposed[t][tt] = anArray[tt][t]
    return transposed


def matrix(random=0, float_random=0, a=1, b=100, square=False):
    if square:
        n = input('Введите размерность матрицы: ')
        while n.isdigit() != 1:
            print("Неверный формат ввода")
            n = input('Введите размерность матрицы: ')
        n = int(n)
        matr = []
        if random == 0:
            for i in range(n):
                t = []
                for j in range(n):
                    _ = input(f'Введите элемент {i + 1} строки {j + 1} столбца: ')
                    while is_number(_) != 1:
                        print("Неверный формат ввода")
                        _ = input(f'Введите элемент {i + 1} строки {j + 1} столбца: ')
                    try:
                        t.append(float(_))
                    except ValueError:
                        try:
                            t.append(complex(_))
                        except ValueError:
                            None
                matr.append(t)
        else:
            for i in range(n):
                t = []
                for j in range(n):
                    if float_random == 1:
                        _ = uniform(a, b)
                        t.append(_)
                    else:
                        _ = randint(a, b)
                        t.append(_)
                matr.append(t)
    else:
        m = input('Введите количество строк: ')

        while m.isdigit() != 1:
            print("Неверный формат ввода")
            m = input('Введите количество строк: ')
        n = input('Введите количество столбцов: ')

        while n.isdigit() != 1:
            print("Неверный формат ввода")
            n = input('Введите количество столбцов: ')
        m = int(m)
        n = int(n)
        matr = []
        if random == 0:
            for i in range(m):
                t = []
                for j in range(n):
                    _ = input(f'Введите элемент {i + 1} строки {j + 1} столбца: ')
                    while is_number(_) != 1:
                        print("Неверный формат ввода")
                        _ = input(f'Введите элемент {i + 1} строки {j + 1} столбца: ')
                    try:
                        t.append(float(_))
                    except ValueError:
                        try:
                            t.append(complex(_))
                        except ValueError:
                            None
                matr.append(t)
        else:
            for i in range(m):
                t = []
                for j in range(n):
                    if float_random == 1:
                        _ = uniform(a, b)
                        t.append(_)
                    else:
                        _ = randint(a, b)
                        t.append(_)
                matr.append(t)

    return matr


# .matrix ( m = int, n = int)
# Возвращает матрицу m x n
# Ввод элементов с клавиатуры
def det2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]


def alg_dop(matrix, somme=None, prod=1):
    if (somme == None):
        somme = []
    if (len(matrix) == 1):
        somme.append(matrix[0][0])
    elif (len(matrix) == 2):
        somme.append(det2(matrix) * prod)
    else:
        for index, elmt in enumerate(matrix[0]):
            transposee = [list(a) for a in zip(*matrix[1:])]
            del transposee[index]
            mineur = [list(a) for a in zip(*transposee)]
            somme = alg_dop(mineur, somme, prod * matrix[0][index] * (-1) ** (index + 2))
    return somme


def determinant(matrix):
    return sum(alg_dop(matrix))


# .determinant(matr = matrix)
# Возвращает определитель matrix


def sum_matrix(mtrx_1, mtrx_2):
    tmp_mtrx = [[0 for j in range(len(mtrx_1))] for i in range(len(mtrx_1[0]))]
    for i in range(len(mtrx_1)):
        for j in range(len(mtrx_1[0])):
            t = type_conversion(mtrx_1[i][j])
            m = type_conversion(mtrx_2[i][j])
            tmp_mtrx[i][j] = t + m
    return tmp_mtrx


def minor(matrix, i, j):
    minor = []
    for q in (matrix[:i] + matrix[i + 1:]):
        _ = q[:j] + q[j + 1:]
        minor.append(_)
    return minor


def subtraction_matrix(mtrx_1, mtrx_2):
    tmp_mtrx = [[0 for j in range(len(mtrx_1))] for i in range(len(mtrx_1[0]))]
    for i in range(len(mtrx_1)):
        for j in range(len(mtrx_1[0])):
            t = type_conversion(mtrx_1[i][j])
            m = type_conversion(mtrx_2[i][j])
            tmp_mtrx[i][j] = t - m
    return tmp_mtrx


def mult_by_count_matrix(mtrx_1, k):
    tmp_mtrx = [[0 for j in range(len(mtrx_1))] for i in range(len(mtrx_1[0]))]
    for i in range(len(mtrx_1)):
        for j in range(len(mtrx_1[0])):
            k = type_conversion(k)
            t = type_conversion(mtrx_1[i][j])
            tmp_mtrx[i][j] = t * k
    return tmp_mtrx


def multiply_matrix(mtrx_1, mtrx_2):
    s = 0
    t = []
    m3 = []
    r1 = len(mtrx_1)
    for z in range(0, r1):
        for j in range(0, r1):
            for i in range(0, r1):
                l1 = type_conversion(mtrx_1[z][i])
                l2 = type_conversion(mtrx_2[i][j])
                s = s + l1 * l2
            t.append(s)
            s = 0
        m3.append(t)
        t = []
    return m3
    return tmp_mtrx


def single_variable(row,
                    index):  # выражение элемента в виде xi = (- a1x1 - a2x2 ... - a(i-1)x(i-1) - a(i+1)x(i+1) ... + с )/ai
    return ([(-i / row[index]) for i in (row[:index] + row[index + 1:-1])] + [row[-1] / row[index]])


def norma(matrix):
    norma_matrix = []
    for i in range(len(matrix)):
        summa = 0
        for j in range(len(matrix)):
            summa += abs(matrix[i][j])
        norma_matrix.append(summa)
    return max(norma_matrix)


def reverse_matrix(matrix):
    deter = determinant(matrix)
    try:
        a = 1 / deter
    except ZeroDivisionError:
        return 'Нулевой определитель'
    matr_dop = [[0] * len(matrix) for i in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matr_dop[i][j] = (-1) ** (i + j) * determinant(minor(matrix, i, j))
    matr_dop_T = matrixTranspose(matr_dop)
    return mult_by_count_matrix(matr_dop_T, a)


def cord(matrix):
    return (norma(matrix) * norma(reverse_matrix(matrix)))


def method_Jacobi(a, b):
    eps = float(input('Введите погрешность для метода Якоби: '))
    matrix = []
    for j in range(len(b)):
        matrix.append(a[j] + b[j])
    interm = [0] * (len(matrix)) + [1]
    variables = [0] * len(matrix)
    k = -1
    interm_2 = [0] * (len(matrix)) + [1]
    count = 0
    while k != 0:
        k = 0
        for i in range(len(matrix)):
            variables[i] = single_variable(matrix[i], i)
            for j in range(len(matrix)):
                ne_know = (interm[:i] + interm[i + 1:])
                interm_2[i] += variables[i][j] * ne_know[j]
            if abs(interm[i] - interm_2[i]) > eps:
                k += 1
        interm = interm_2
        interm_2 = [0] * (len(matrix)) + [1]
        # print(interm[:-1])
        # print(k)
        # print('____')
        count += 1
        if count == 1000:
            return (['Метод Якоби не сработал'] * 3)
    return (a, reverse_matrix(a), interm[:-1])


def pick_nonzero_row(m, k):
    while k < m.shape[0] and not m[k, k]:
        k += 1
    return k


def gssjrdn(a, b):
    nc = []
    for i in range(len(a)):
        nc.append(a[i])
    a = np.array(a, float)
    b = np.array(b, float)
    n = len(b)
    st = a

    m = np.hstack((st,
                   np.matrix(np.diag([1.0 for i in range(st.shape[0])]))))
    for k in range(n):
        swap_row = pick_nonzero_row(m, k)
        if swap_row != k:
            m[k, :], m[swap_row, :] = m[swap_row, :], np.copy(m[k, :])
        if m[k, k] != 1:
            m[k, :] *= 1 / m[k, k]
        for row in range(k + 1, n):
            m[row, :] -= m[k, :] * m[row, k]
    for k in range(n - 1, 0, -1):
        for row in range(k - 1, -1, -1):
            if m[row, k]:
                m[row, :] -= m[k, :] * m[row, k]

    for k in range(n):
        if np.fabs(a[k, k]) < 1.0e-12:
            for i in range(k + 1, n):
                if np.fabs(a[i, k]) > np.fabs(a[k, k]):
                    for j in range(k, n):
                        a[k, j], a[i, j] = a[i, j], a[k, j]
                    b[k], b[i] = b[i], b[k]
                    break
        pivot = a[k, k]
        for j in range(k, n):
            a[k, j] /= pivot
        b[k] /= pivot
        for i in range(n):
            if i == k or a[i, k] == 0:
                continue
            factor = a[i, k]
            for j in range(k, n):
                a[i, j] -= factor * a[k, j]
            b[i] -= factor * b[k]

    return nc, np.hsplit(m, n // 2)[0], b


def frkgssjrdn(a, b):
    nc = []
    for i in range(len(a)):
        nc.append(a[i])
    a = np.array(a, float)
    b = np.array(b, float)
    n = len(b)

    for i in range(n):
        for j in range(n):
            a[i, j] = Fraction(a[i, j])
            b[i] = Fraction(*b[i])

    matrix = []
    for j in range(n):
        matrix.append(a[j] + b[j])
    matrix = np.array(matrix, float)
    matrix[i, j] = Fraction(matrix[i, j])

    for k in range(n):
        if np.fabs(a[k, k]) < 1.0e-12:
            for i in range(k + 1, n):
                if np.fabs(a[i, k]) > np.fabs(a[k, k]):
                    for j in range(k, n):
                        a[k, j], a[i, j] = a[i, j], a[k, j]
                    b[k], b[i] = b[i], b[k]
                    break
        pivot = a[k, k]
        for j in range(k, n):
            a[k, j] /= pivot
        b[k] /= pivot
        for i in range(n):
            if i == k or a[i, k] == 0:
                continue
            factor = a[i, k]
            for j in range(k, n):
                a[i, j] -= factor * a[k, j]
            b[i] -= factor * b[k]

    m = np.hstack((matrix,
                   np.matrix(np.diag([1.0 for i in range(matrix.shape[0])]))))
    for k in range(n):
        swap_row = pick_nonzero_row(m, k)
        if swap_row != k:
            m[k, :], m[swap_row, :] = m[swap_row, :], np.copy(m[k, :])
        if m[k, k] != 1:
            m[k, :] *= 1 / m[k, k]
        for row in range(k + 1, n):
            m[row, :] -= m[k, :] * m[row, k]
    for k in range(n - 1, 0, -1):
        for row in range(k - 1, -1, -1):
            if m[row, k]:
                m[row, :] -= m[k, :] * m[row, k]
    return nc, np.hsplit(m, n // 2)[0], b


def method_lin(x, y):
    x_1 = 0
    x_2 = 0
    x_3 = 0
    x_4 = 0
    x2_y = 0
    x_y = 0
    y_1 = 0
    for i in range(len(x)):
        x_1 += x[i]
        x_2 += x[i] ** 2
        x_3 += x[i] ** 3
        x_4 += x[i] ** 4
        x2_y += y[i] * x[i] ** 2
        x_y += y[i] * x[i]
        y_1 += y[i]
    n = len(x)
    a = [[x_2, x_3, x_4], [x_1, x_2, x_3], [n, x_1, x_2]]
    b = [[x2_y], [x_y], [y_1]]
    roots = gssjrdn(a, b)[2]
    c = []
    for i in range(2):
        c.append(*roots[i])

    def f_x(t):
        return c[1] * t + c[0]

    gamma = 0
    f = []
    for i in range(len(x)):
        f.append(f_x(x[i]))
        gamma += (y[i] - f[i]) ** 2
    exp = ''
    for i in [1, 0]:
        if c[i] != 0:
            exp += f'{c[i]}*t**{i} + '
    exp = exp[:-2]
    output = [[x[i]] + [y[i]] + [f[i]] for i in range(len(x))]
    return (output, exp, gamma)


def method_min_square(x, y):
    x_1 = 0
    x_2 = 0
    x_3 = 0
    x_4 = 0
    x2_y = 0
    x_y = 0
    y_1 = 0
    for i in range(len(x)):
        x_1 += x[i]
        x_2 += x[i] ** 2
        x_3 += x[i] ** 3
        x_4 += x[i] ** 4
        x2_y += y[i] * x[i] ** 2
        x_y += y[i] * x[i]
        y_1 += y[i]
    n = len(x)
    a = [
        [x_2, x_3, x_4],
        [x_1, x_2, x_3],
        [n, x_1, x_2]
    ]
    b = [
        [x2_y],
        [x_y],
        [y_1]
    ]
    roots = gssjrdn(a, b)[2]
    c = []
    for i in range(3):
        c.append(*roots[i])

    def f_x(t):
        return c[2] * t ** 2 + c[1] * t + c[0]

    gamma = 0
    f = []
    for i in range(len(x)):
        f.append(f_x(x[i]))
        gamma += (y[i] - f[i]) ** 2
    exp = ''
    for i in [2, 1, 0]:
        if c[i] != 0:
            exp += f'{c[i]}*t**{i} + '
    exp = exp[:-2]
    output = [[x[i]] + [y[i]] + [f[i]] for i in range(len(x))]
    return (output, exp, gamma)


def lagranz(x, y):
    t = symbols('t')
    z = 0
    for j in range(len(y)):
        numerator = 1;
        denominator = 1;
        for i in range(len(x)):
            if i == j:
                numerator = numerator * 1;
                denominator = denominator * 1
            else:
                numerator = expand(numerator * (t - x[i]))
                denominator = denominator * (x[j] - x[i])
        z = expand(z + y[j] * numerator / denominator)
    f_x = lambdify(t, z)
    output = []
    for k in range(len(x)):
        output.append([x[k], y[k], f_x(x[k])])
    return (output, z)


# Вывод в формате ([[x1,y1,f_x1]...[xi,yi,f_xi]], f_x)
# f_x - строка, можно преобразовать в функцию типа f_x(x0) = y0

class SplineTuple:
    def __init__(self, a, b, c, d, x):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x = x


# Построение сплайна
# x - узлы сетки, должны быть упорядочены по возрастанию, кратные узлы запрещены
# y - значения функции в узлах сетки
# n - количество узлов сетки
def BuildSpline(x, y):
    # Инициализация массива сплайнов
    n = len(x)
    splines = [SplineTuple(0, 0, 0, 0, 0) for _ in range(0, n)]
    for i in range(0, n):
        splines[i].x = x[i]
        splines[i].a = y[i]

    splines[0].c = splines[n - 1].c = 0.0

    # Решение СЛАУ относительно коэффициентов сплайнов c[i] методом прогонки для трехдиагональных матриц
    # Вычисление прогоночных коэффициентов - прямой ход метода прогонки
    alpha = [0.0 for _ in range(0, n - 1)]
    beta = [0.0 for _ in range(0, n - 1)]

    for i in range(1, n - 1):
        hi = x[i] - x[i - 1]
        hi1 = x[i + 1] - x[i]
        A = hi
        C = 2.0 * (hi + hi1)
        B = hi1
        F = 6.0 * ((y[i + 1] - y[i]) / hi1 - (y[i] - y[i - 1]) / hi)
        z = (A * alpha[i - 1] + C)
        alpha[i] = -B / z
        beta[i] = (F - A * beta[i - 1]) / z

    # Нахождение решения - обратный ход метода прогонки
    for i in range(n - 2, 0, -1):
        splines[i].c = alpha[i] * splines[i + 1].c + beta[i]

    # По известным коэффициентам c[i] находим значения b[i] и d[i]
    for i in range(n - 1, 0, -1):
        hi = x[i] - x[i - 1]
        splines[i].d = (splines[i].c - splines[i - 1].c) / hi
        splines[i].b = hi * (2.0 * splines[i].c + splines[i - 1].c) / 6.0 + (y[i] - y[i - 1]) / hi
    return splines


# Вычисление значения интерполированной функции в произвольной точке
def Interpolate(splines, x):
    if not splines:
        return None  # Если сплайны ещё не построены - возвращаем NaN

    n = len(splines)
    s = SplineTuple(0, 0, 0, 0, 0)

    if x <= splines[0].x:  # Если x меньше точки сетки x[0] - пользуемся первым эл-тов массива
        s = splines[0]
    elif x >= splines[n - 1].x:  # Если x больше точки сетки x[n - 1] - пользуемся последним эл-том массива
        s = splines[n - 1]
    else:  # Иначе x лежит между граничными точками сетки - производим бинарный поиск нужного эл-та массива
        i = 0
        j = n - 1
        while i + 1 < j:
            k = i + (j - i) // 2
            if x <= splines[k].x:
                j = k
            else:
                i = k
        s = splines[j]

    dx = x - s.x
    return s.a + (s.b + (s.c / 2.0 + s.d * dx / 6.0) * dx) * dx;


accuracy = 0.00001
START_X = -1
END_X = 6
START_Y = -1
END_Y = 20
temp = []


def whence_differences(y_array):
    return_array = []
    for i in range(0, len(y_array) - 1):
        return_array.append(y_array[i + 1] - y_array[i])
    return return_array


def witchcraft_start(y_array, h):
    part_y = [y_array[0]]
    y = y_array
    for i in range(0, len(y_array) - 1):
        y = whence_differences(y)
        part_y.append(y[0] / math.factorial(i + 1) / (h ** (i + 1)))
    return part_y


def tragic_magic(coefficients_y, point, x_array):
    value = coefficients_y[0]
    for i in range(1, len(coefficients_y)):
        q = 1
        for j in range(0, i):
            q *= (point - x_array[j])
        value += coefficients_y[i] * q
    return value


def build_points(x_array, y_array):
    for i in range(0, len(x_array)):
        plt.scatter(x_array[i], y_array[i])


def newton_there(x_array, y_array):
    x0 = x_array[0]
    h = x_array[1] - x_array[0]
    build_points(x_array, y_array)
    part_y = witchcraft_start(y_array, h)
    x = np.linspace(x_array[0], x_array[len(x_array) - 1], 228)
    return (x, tragic_magic(part_y, x, x_array), part_y)


def witchcraft_continue(y_array, h):
    part_y = [y_array[len(y_array) - 1]]
    y = y_array
    for i in range(0, len(y_array) - 1):
        y = whence_differences(y)
        part_y.append(y[len(y) - 1] / math.factorial(i + 1) / (h ** (i + 1)))
    return part_y


def ecstatic_magic(coefficients_y, point, x_array):
    value = coefficients_y[0]
    for i in range(1, len(coefficients_y)):
        q = 1
        for j in range(0, i):
            q *= (point - x_array[len(x_array) - j - 1])
        value += coefficients_y[i] * q
    return value


def newton_here_the_boss(x_array, y_array):
    x0 = x_array[0]
    h = x_array[1] - x_array[0]
    build_points(x_array, y_array)
    part_y = witchcraft_continue(y_array, h)
    x = np.linspace(x_array[0], x_array[len(x_array) - 1], 228)
    return (x, ecstatic_magic(part_y, x, x_array), part_y)


def approximate_log_function(x, y):
    C = np.arange(0.01, 1, step=0.01)
    a = np.arange(0.01, 1, step=0.01)
    b = np.arange(0.01, 1, step=0.01)

    min_mse = 9999999999
    parameters = [0, 0, 0]

    for i in np.array(np.meshgrid(C, a, b)).T.reshape(-1, 3):
        y_estimation = i[0] * np.log(i[1] * np.array(x) + i[2])
        mse = mean_squared_error(y, y_estimation)
        if mse < min_mse:
            min_mse = mse
            parameters = [i[0], i[1], i[2]]
    output = [[x[i]] + [y[i]] + [y_estimation[i]] for i in range(len(x))]
    return (output, min_mse)


def ap_norm_rasp(x, y_real):
    def norm(x, mean, sd):
        norm = []
        for i in range(len(x)):
            norm += [1.0 / (sd * np.sqrt(2 * np.pi)) * np.exp(-(x[i] - mean) ** 2 / (2 * sd ** 2))]
        return np.array(norm)

    mean1, mean2 = 10, -2
    std1, std2 = 0.5, 10
    m, dm, sd1, sd2 = [3, 10, 1, 1]
    p = [m, dm, sd1, sd2]  # Initial guesses for leastsq
    y_init = norm(x, m, sd1) + norm(x, m + dm, sd2)  # For final comparison plot

    def res(p, y, x):
        m, dm, sd1, sd2 = p
        m1 = m
        m2 = m1 + dm
        y_fit = norm(x, m1, sd1) + norm(x, m2, sd2)
        err = y - y_fit
        return err

    plsq = leastsq(res, p, args=(y_real, x))
    y_est = norm(x, plsq[0][0], plsq[0][2]) + norm(x, plsq[0][0] + plsq[0][1], plsq[0][3])
    gamma = 0
    for i in range(len(x)):
        gamma += (y_real[i] - y_init[i]) ** 2
        output = [[x[i]] + [y_real[i]] + [y_init[i]] for i in range(len(x))]
    return (output, gamma)


def approximate_exp_function(x, y):
    x_1 = 0
    x_2 = 0
    x_3 = 0
    x_4 = 0
    x2_y = 0
    x_y = 0
    y_1 = 0
    for i in range(len(x)):
        x_1 += x[i]
        x_2 += x[i] ** 2
        x_3 += x[i] ** 3
        x_4 += x[i] ** 4
        x2_y += y[i] * x[i] ** 2
        x_y += y[i] * x[i]
        y_1 += y[i]
    n = len(x)
    a = [[x_2, x_3, x_4], [x_1, x_2, x_3], [n, x_1, x_2]]
    b = [[x2_y], [x_y], [y_1]]
    roots = gssjrdn(a, b)[2]
    c = []
    for i in range(3):
        c.append(*roots[i])

    def f_x(t):
        return c[0] * math.e ** (c[1] * t)

    gamma = 0
    f = []
    for i in range(len(x)):
        f.append(f_x(x[i]))
        gamma += (y[i] - f[i]) ** 2
    exp = ''
    for i in [1, 0]:
        if c[i] != 0:
            exp += f'{c[i]}*e**({c[i + 1]}*t) + '
    exp = exp[:-2]
    output = [[x[i]] + [y[i]] + [f[i]] for i in range(len(x))]
    return (output, exp, gamma)


def interpol_st(xp, fp):
    x = np.linspace(-np.pi, 10, 100)
    y = np.interp(x, xp, fp)
    fig, ax = plt.subplots()
    plt.plot(xp, fp, 'b', label="Исходные точки")
    plt.plot(x, y, 'r', label=f'Интерполированная функция')
    plt.title('Интерполяция точек стандартной numpy')
    plt.show()


def aprox_st(xp, fp):
    x = np.linspace(-len(xp) / 2, len(xp) / 2, 100)
    y = np.polyfit(xp, fp, 99)
    fig, ax = plt.subplots()
    plt.plot(xp, fp, 'b', label="Исходные точки")
    plt.plot(x, y, 'r', label=f'Интерполированная функция')
    plt.title('Апроксимация точек стандартной numpy')
    plt.show()


def add_edge(f_item, s_item, graph=None):
    graph.add_edge(f_item, s_item)
    graph.add_edge(s_item, f_item)


def method_ant(distmat, itermax, alpha=0.6, beta=0.65, pheEvaRate=0.3):
    def lengthCal(antPath, distmat):  # Рассчитать расстояние
        length = []
        dis = 0
        for i in range(len(antPath)):
            for j in range(len(antPath[i]) - 1):
                dis += distmat[antPath[i][j]][antPath[i][j + 1]]
            dis += distmat[antPath[i][-1]][antPath[i][0]]
            length.append(dis)
            dis = 0
        return length
    x_plot = []
    y_plot = []
    cost_min = 9999999
    cityNum = len(distmat)  # Число городов
    antNum = cityNum  # Число муравьев
    pheromone = np.ones((cityNum, cityNum))  # Феромоновая матрица
    heuristic = 1 / (np.eye(cityNum) + distmat) - np.eye(cityNum)
    iter = 1  # Итерации
    antPath = np.zeros((antNum, cityNum)).astype(int) - 1
    while iter < itermax:
        antPath = []
        for i in range(cityNum):
            antPath.append([-1] * cityNum)
        firstCity = [i for i in range(cityNum)]
        rd.shuffle(firstCity)  # Случайно назначьте начальный город для каждого муравья
        unvisted = []
        p = []
        pAccum = 0
        for i in range(len(antPath)):
            antPath[i][0] = firstCity[i]
        for i in range(
                len(antPath[0]) - 1):  # Постепенно обновляйте следующий город, в который собирается каждый муравей
            for j in range(len(antPath)):
                for k in range(cityNum):
                    if k not in antPath[j]:
                        unvisted.append(k)
                for m in unvisted:
                    pAccum += pheromone[antPath[j][i]][m] ** alpha * heuristic[antPath[j][i]][m] ** beta
                for n in unvisted:
                    p.append(pheromone[antPath[j][i]][n] ** alpha * heuristic[antPath[j][i]][n] ** beta / pAccum)
                roulette = np.array(p).cumsum()  # Создать рулетку
                r = rd.uniform(min(roulette), max(roulette))
                for x in range(len(roulette)):
                    if roulette[x] >= r:  # Используйте метод рулетки, чтобы выбрать следующий город
                        antPath[j][i + 1] = unvisted[x]
                        break
                unvisted = []
                p = []
                pAccum = 0
        pheromone = (1 - pheEvaRate) * pheromone  # Феромон летучий
        length = lengthCal(antPath, distmat)
        for i in range(len(antPath)):
            for j in range(len(antPath[i]) - 1):
                pheromone[antPath[i][j]][antPath[i][j + 1]] += 1 / length[i]  # Обновление феромона
            pheromone[antPath[i][-1]][antPath[i][0]] += 1 / length[i]

        # Визуализация
        # graph = nx.Graph()
        # for i in range(len(antPath[length.index(min(length))]) - 1):
        # add_edge(antPath[length.index(min(length))][i], antPath[length.index(min(length))][i + 1], graph=graph)
        # nx.draw_circular(graph,
        # node_color='red',
        # node_size=1000,
        # with_labels=True)
        # plt.title(f"Итерация {iter}")
        # plt.show()
        if min(length) < cost_min:
            cost_min = min(length)
            road_min = antPath[length.index(min(length))] + [antPath[length.index(min(length))][0]]
        x_plot.append(iter)
        y_plot.append(cost_min)
        iter += 1
    plt.plot(x_plot,y_plot)
    plt.show()
    return (cost_min, road_min)


def TSP_SA(G):
    s = list(range(len(G)))
    c = cost(G, s)
    ntrial = 1
    T = 30
    alpha = 0.99
    while ntrial <= 1000:
        n = np.random.randint(0, len(G))
        while True:
            m = np.random.randint(0, len(G))
            if n != m:
                break
        s1 = swap(s, m, n)
        c1 = cost(G, s1)
        if c1 < c:
            s, c = s1, c1
        else:
            if np.random.rand() < np.exp(-(c1 - c) / T):
                s, c = s1, c1
        T = alpha * T
        ntrial += 1
    s = s + [s[0]]
    return (s, c)


def swap(s, m, n):
    i, j = min(m, n), max(m, n)
    s1 = s.copy()
    while i < j:
        s1[i], s1[j] = s1[j], s1[i]
        i += 1
        j -= 1
    return s1


def cost(G, s):
    l = 0
    for i in range(len(s) - 1):
        l += G[s[i]][s[i + 1]]
    l += G[s[len(s) - 1]][s[0]]
    return l


def copyToFinal(curr_path, N):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]


def firstMin(adj, i, N):
    min_ = maxsize
    for k in range(N):
        if adj[i][k] < min_ and i != k:
            min_ = adj[i][k]
    return min_


def secondMin(adj, i, N):
    first, second = maxsize, maxsize
    for j in range(N):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]
        elif (adj[i][j] <= second and
              adj[i][j] != first):
            second = adj[i][j]
    return second


def TSPRec(adj, curr_bound, curr_weight, level, curr_path, visited, N):
    global final_res
    if level == N:
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
            if curr_res < final_res:
                copyToFinal(curr_path, N)
                final_res = curr_res
        return

    for i in range(N):
        if (adj[curr_path[level - 1]][i] != 0 and visited[i] == False):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]
            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1], N) + firstMin(adj, i, N)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1], N) + firstMin(adj, i, N)) / 2)
            if curr_bound + curr_weight < final_res:
                curr_path[level] = i
                visited[i] = True
                TSPRec(adj, curr_bound, curr_weight, level + 1, curr_path, visited, N)
            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp
            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True


def TSP(adj, N):
    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N
    for i in range(N):
        curr_bound += (firstMin(adj, i, N) + secondMin(adj, i, N))
    curr_bound = math.ceil(curr_bound / 2)
    visited[0] = True
    curr_path[0] = 0
    TSPRec(adj, curr_bound, 0, 1, curr_path, visited, N)



N = int(input("Число вершин: "))
adj = matrix(random = 1, square= True)
final_res = float('inf')
final_path = [None] * (N + 1)
visited = [False] * N

itermax = 100
start_time = time.time()
print("Алгоритм: Метод ветвей и границ")
print(f"Количество итераций: {itermax}")
TSP(adj,N)
print(f"Время работы алгоритма: {time.time() - start_time} секунд")
print("Полученный путь : ", end = ' ')
print(final_path)
print("Стоимость :", final_res)

itermax = 100
start_time = time.time()
print("Алгоритм: Имитация отжига")
print(f"Количество итераций: {itermax}")
print(f"Время работы алгоритма: {time.time() - start_time} секунд")
print(f"Полученный путь: {TSP_SA(adj)[0]}")
print(f"Стоимость: {TSP_SA(adj)[1]}")

itermax = 100
start_time = time.time()
output = method_ant(distmat=adj, itermax= itermax )
print('Алгоритм муравьиной колонии')
print(f'Количество итераций: {itermax}')
print('Время выполнения методом муравьиной колонии:')
print("-— %s seconds —-" % (time.time() - start_time))
print('Результат: ' ,output[1])
print('Стоимость: ', output[0])






