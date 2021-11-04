from matrix import *
import numpy as np
from fractions import Fraction

#Итерационный метод


def single_variable(row,index): #выражение элемента в виде xi = (c - a1x1 - a2x2 ... - a(i-1)x(i-1) - a(i+1)x(i+1) ...)/ai
    return ([row[-1]/row[index]] + [(-i/row[index]) for i in (row[:index] + row[index+1:-1]) ])


def method_Jacobi(matrix):
    eps = float(input('Введите погрешность: '))
    interm = [1] + [0] * (len(matrix))
    b = [0] * len(matrix)
    k = -1
    interm_2 = [1] + [0] * (len(matrix))
    while k != 0:
        k = 0
        for i in range(len(matrix)):
            b[i] = single_variable(matrix[i], i)
            for j in range(len(matrix)):
                _ = (interm[:i + 1] + interm[i + 2:])
                interm_2[i + 1] += b[i][j] * _[j]
            if abs(interm[i + 1] - interm_2[i + 1]) > eps:
                k += 1
        interm = interm_2
        interm_2 = [1] + [0] * (len(matrix))
    return interm


method_Jacobi(matrix())

#Начал считать в 06:12
#Надеюсь посчитает