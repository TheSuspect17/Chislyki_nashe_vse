from random import uniform, randint
from fractions import Fraction
import numpy as np


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

def type_conversion(str): #преобразование типов
    try:
        float(str)
        return float(str)
    except ValueError:
        complex(str)
        return complex(str)


# .type_conversion(str = string)
# Возвращает float(str), при ошибке типа ValueError complex(str)

def matrixTranspose(anArray):
    transposed = [None]*len(anArray[0])
    for t in range(len(anArray)):
        transposed[t] = [None]*len(anArray)
        for tt in range(len(anArray[t])):
            transposed[t][tt] = anArray[tt][t]
    return transposed

def matrix(random = 0, float_random = 0, a = 1, b = 100):
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
                    _ = uniform(a,b)
                    t.append(_)
                else:
                    _ = randint(a,b)
                    t.append(_)
            matr.append(t)

    return matr

#.matrix ( m = int, n = int)
# Возвращает матрицу m x n
# Ввод элементов с клавиатуры
def det2(matrix):
    return matrix[0][0]*matrix[1][1]-matrix[1][0]*matrix[0][1]

def alg_dop(matrix,somme=None,prod=1):
    if(somme==None):
        somme=[]
    if(len(matrix)==1):
        somme.append(matrix[0][0])
    elif (len(matrix) == 2):
        somme.append(det2(matrix) * prod)
    else:
        for index, elmt in enumerate(matrix[0]):
            transposee = [list(a) for a in zip(*matrix[1:])]
            del transposee[index]
            mineur = [list(a) for a in zip(*transposee)]
            somme = alg_dop(mineur,somme,prod*matrix[0][index]*(-1)**(index+2))
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



def minor(matrix, i,j):
    minor = []
    for q in (matrix[:i] + matrix[i+1:]):
        _ = q[:j]+q[j+1:]
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



def single_variable(row,index): #выражение элемента в виде xi = (- a1x1 - a2x2 ... - a(i-1)x(i-1) - a(i+1)x(i+1) ... + с )/ai
    return ([(-i/row[index]) for i in (row[:index] + row[index+1:-1]) ] + [row[-1]/row[index]])

def method_Jacobi(a,b):
    eps = float(input('Введите погрешность: '))
    matrix = []
    for j in range(len(b)):
        matrix.append(a[j]+b[j])
    interm =  [0] * (len(matrix)) + [1]
    variables = [0] * len(matrix)
    k = -1
    interm_2 =  [0] * (len(matrix)) + [1]
    while k != 0:
        k = 0
        for i in range(len(matrix)):
            variables[i] = single_variable(matrix[i], i)
            for j in range(len(matrix)):
                ne_eby = (interm[:i ] + interm[i + 1:])
                interm_2[i] += variables[i][j] * ne_eby[j]
            if abs(interm[i] - interm_2[i]) > eps:
                k += 1
        interm = interm_2
        interm_2 =  [0] * (len(matrix)) + [1]
        #print(interm[:-1])
        #print(k)
        #print('____')
    return interm[:-1]

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
        a = 1/deter
    except ZeroDivisionError:
        return 'Быдло'
    matr_dop = [[0]*len(matrix) for i in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matr_dop[i][j] = (-1)**(i+j)*determinant(minor(matrix,i,j))
    matr_dop_T = matrixTranspose(matr_dop)
    return mult_by_count_matrix(matr_dop_T,a)


def cord(matrix):
    return (norma(matrix)*norma(reverse_matrix(matrix)))

def pick_nonzero_row(m, k):
    while k < m.shape[0] and not m[k, k]:
        k += 1
    return k

def gssjrdn(a, b):
    nc = []
    for i in range (len(a)):
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
        if np.fabs(a[k,k]) < 1.0e-12:
            for i in range(k+1,n):
                if np.fabs(a[i,k]) > np.fabs(a[k,k]):
                    for j in range (k,n):
                        a[k,j], a[i,j] = a[i,j], a[k,j]
                    b[k], b[i] = b[i], b[k]
                    break
        pivot = a[k,k]
        for j in range (k,n):
            a[k,j] /= pivot
        b[k] /=pivot
        for i in range(n):
            if i == k or a[i,k] ==0: 
                continue
            factor = a[i,k]
            for j in range (k,n):
                a[i,j] -= factor * a[k,j]
            b[i] -= factor * b[k]
      
    return nc, np.hsplit(m, n // 2)[1], b

def frkgssjrdn(a, b):
    nc = []
    for i in range (len(a)):
        nc.append(a[i])
    a = np.array(a, float)
    b = np.array(b, float)
    n = len(b)
    
    for i in range (n):
        for j in range(n):
            a[i,j] = Fraction(a[i,j])
            b[i] = Fraction(b[i])
    
    matrix = []
    for j in range(n):
        matrix.append(a[j]+b[j])
    matrix = np.array(matrix, float)
    matrix[i,j] = Fraction(matrix[i,j])
    
    for k in range(n):
        if np.fabs(a[k,k]) < 1.0e-12:
            for i in range(k+1,n):
                if np.fabs(a[i,k]) > np.fabs(a[k,k]):
                    for j in range (k,n):
                        a[k,j], a[i,j] = a[i,j], a[k,j]
                    b[k], b[i] = b[i], b[k]
                    break
        pivot = a[k,k]
        for j in range (k,n):
            a[k,j] /= pivot
        b[k] /=pivot
        for i in range(n):
            if i == k or a[i,k] ==0: 
                continue
            factor = a[i,k]
            for j in range (k,n):
                a[i,j] -= factor * a[k,j]
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
    return nc, np.hsplit(m, n // 2)[1],b       
