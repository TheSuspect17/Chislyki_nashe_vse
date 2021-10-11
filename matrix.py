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
# .is_number ( str = string )
# return: TRUE если str может быть преобразован к типу complex или float

def type_conversion(str): #преобразование типов
    try:
        float(str)
        return float(str)
    except ValueError:
        complex(str)
        return complex(str)

# .type_conversion(str = string)
# Возвращает float(str), при ошибке типа ValueError complex(str)

def matrix(m,n):
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
    return matr

#.matrix ( m = int, n = int)
# Возвращает матрицу m x n
# Ввод элементов с клавиатуры

def determinant(matr):
    size = len(matr)
    if size == 1:
        return matr[0][0]
    for k in range(size):
        t = [ row[:k] + row[k+1:] for row in (matr[1:])]
        matr[0][k] = matr[0][k] * (-1)**(k) * determinant(t)
    return sum(matr[0])

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


