import numpy as np
import time
import random as rd
from numpy import linalg as LA

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def determinant(matr,size):
    #print('Произошла рекурсия')
    if size == 1:
        return matr[0][0]
    for k in range(size):
        #print(f"Работаем с {k} элементом строки")
        t = getMatrixMinor(matr,0,k)
        #print(t)
        matr[0][k] = matr[0][k] * (-1)**(k) * determinant(t,size-1)
    return sum(matr[0])


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

m = input('Введите размер матрицы: ')

while m.isdigit() != 1:
    print("Неверный формат ввода")
    m = input('Введите количество строк: ')



m = int(m)
n = int(m)
matr = []
for i in range(m):
    t = []
    for j in range(n):
        #_ = input(f'Введите элемент {i + 1} строки {j + 1} столбца: ')
        _ = rd.randint(0,100)
        #_ = 0
        while is_number(_) != 1:
            print("Неверный формат ввода")
            _ = input(f'Введите элемент {i + 1} строки {j + 1} столбца: ')
        # t.append(float(_))
        try:
            t.append(float(_))
        except ValueError:
            try:
                t.append(complex(_))
            except ValueError:
                None
    matr.append(t)

start_time = time.perf_counter()

#for i in range(len(matr)):
#    print(matr[i])



print(determinant(matr,n))
#print(LA.det(np.matrix((matr))))

print("-— %s seconds —-" % (time.perf_counter() - start_time))
