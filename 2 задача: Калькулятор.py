import numpy as np
import sympy
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
def type_conversion(str): #преобразование типов
    try:
        float(str)
        return float(str)
    except ValueError:
        complex(str)
        return complex(str)

def schit():
    m = input('Введите число строк матрицы: ')
    while not m.isdigit():
        print("Вы ввели некорректное число строк, повторите попытку: ")
        m = input('Введите число строк матрицы: ')
        if m.isdigit() and int(m) == 0:
            m = '*'
    n = input('Введите число столбцов матрицы: ')
    while not n.isdigit() :
        print("Вы ввели некорректное число столбцов, повторите попытку: ")
        n = input('Введите число столбцов матрицы: ')
        if n.isdigit() and int(n) == 0:
            n = '*'


    m = int(m)
    n = int(n)

    mtrx = [[0 for j in range(n)] for i in range(m)]

    for i in range(m):
        for j in range(n):
            k = input(f'Введите элемент {i + 1} строки {j + 1} столбца: ')
            while is_number(k) != 1:
                print("Неверный формат ввода")
                k = input(f'Введите элемент {i + 1} строки {j + 1} столбца: ')
            mtrx[i][j] = k
    return mtrx


def summ(mtrx_1, mtrx_2):
    tmp_mtrx = [[0 for j in range(len(mtrx_1))] for i in range(len(mtrx_1[0]))]
    for i in range(len(mtrx_1)):
        for j in range(len(mtrx_1[0])):
            t = type_conversion(mtrx_1[i][j])
            m = type_conversion(mtrx_2[i][j])
            tmp_mtrx[i][j] = t + m
    return tmp_mtrx


def vychet(mtrx_1, mtrx_2):
    tmp_mtrx = [[0 for j in range(len(mtrx_1))] for i in range(len(mtrx_1[0]))]
    for i in range(len(mtrx_1)):
        for j in range(len(mtrx_1[0])):
            t = type_conversion(mtrx_1[i][j])
            m = type_conversion(mtrx_2[i][j])
            tmp_mtrx[i][j] = t - m
    return tmp_mtrx


def mult_by_count(mtrx_1, k):
    tmp_mtrx = [[0 for j in range(len(mtrx_1))] for i in range(len(mtrx_1[0]))]
    for i in range(len(mtrx_1)):
        for j in range(len(mtrx_1[0])):
            k = type_conversion(k)
            t = type_conversion(mtrx_1[i][j])
            tmp_mtrx[i][j] = t * k


def mult(mtrx_1, mtrx_2):
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


B = 0
while True:
    print("Что будем делать?")
    print("0.Выйти из калькулятора")
    print("1.Сложение матриц ")
    print("2.Вычитание матриц ")
    print("3.Умножение матрицы на число ")
    print("4.Умножение матриц ")
    choise_1 = input('Сделайте ваш выбор нажав [0/1/2/3/4] : ')
    while choise_1 not in ["0","1","2","3","4"]:
        print('Неверный формат ввода: ')
        choise_1 = input('Сделайте ваш выбор нажав [0/1/2/3/4] : ')
    choise_1 = int(choise_1)

    if choise_1 == 0:
        break
    if choise_1 == 1:
        print(f'На данный момент в буffере находится матрица {B}, если вы хотите ее использовать для вычислений?')
        print("0.Нет")
        print("1.Да ")
        choise_2 = input('Сделайте ваш выбор нажав [0/1] :')
        while choise_2 not in ["0", "1"]:
            print('Неверный формат ввода: ')
            choise_2 = input('Сделайте ваш выбор нажав [0/1] :')
        choise_2 = int(choise_2)

        if choise_2 == 0:
            print('Введите 1 матрицу: ')
            mtrx_1 = schit()
        else:
            mtrx_1 = B
        print('Введите 2 матрицу: ')
        mtrx_2 = schit()

        print('Получаем: ')
        B = summ(mtrx_1, mtrx_2)
        print(f'{mtrx_1} + {mtrx_2} = {summ(mtrx_1, mtrx_2)}')
    if choise_1 == 2:
        print(f'На данный момент в буffере находится матрица {B}, если вы хотите ее использовать для вычислений?')
        print("0.Нет")
        print("1.Да ")
        choise_2 = input('Сделайте ваш выбор нажав [0/1] :')
        while choise_2 not in ["0", "1"]:
            print('Неверный формат ввода: ')
            choise_2 = input('Сделайте ваш выбор нажав [0/1] :')
        choise_2 = int(choise_2)
        if choise_2 == 0:
            print('Введите 1 матрицу: ')
            mtrx_1 = schit()
        else:
            mtrx_1 = B
        print('Введите 2 матрицу: ')
        mtrx_2 = schit()
        print('Получаем: ')
        B = vychet(mtrx_1, mtrx_2)
        print(f'{mtrx_1} - {mtrx_2} = {vychet(mtrx_1, mtrx_2)}')
    if choise_1 == 3:
        print(f'На данный момент в буffере находится матрица {B}, если вы хотите ее использовать для вычислений?')
        print("0.Нет")
        print("1.Да ")
        choise_2 = input('Сделайте ваш выбор нажав [0/1] :')
        while choise_2 not in ["0", "1"]:
            print('Неверный формат ввода: ')
            choise_2 = input('Сделайте ваш выбор нажав [0/1] :')
        choise_2 = int(choise_2)
        if choise_2 == 0:
            print('Введите 1 матрицу: ')
            mtrx_1 = schit()
        else:
            mtrx_1 = B
        k = input('Введите число: ')
        while not is_number(k):
            print('Неверный формат ввода: ')
            k = input('Введите число: ')

        k = type_conversion(k)
        B = mult_by_count(mtrx_1, k)
        print(f'{mtrx_1} * {k} = {mult_by_count(mtrx_1, k)}')
    if choise_1 == 4:
        print(f'На данный момент в буffере находится матрица {B}, если вы хотите ее использовать для вычислений?')
        print("0.Нет")
        print("1.Да ")
        choise_2 = input('Сделайте ваш выбор нажав [0/1] :')
        while choise_2 not in ["0", "1"]:
            print('Неверный формат ввода: ')
            choise_2 = input('Сделайте ваш выбор нажав [0/1] :')
        choise_2 = int(choise_2)
        if choise_2 == 0:
            print('Введите 1 матрицу: ')
            mtrx_1 = schit()
        else:
            mtrx_1 = B
        print('Введите 2 матрицу: ')
        mtrx_2 = schit()
        print('Получаем: ')
        B = mult(mtrx_1, mtrx_2)
        print(f'{mtrx_1} * {mtrx_2} = {mult(mtrx_1, mtrx_2)}')
    print('Результат сохранен в буffер.')
