from matrix import *

print('Осуществляется ввод коэффициентов: ')
matrix_сoef = matrix(random=1) # Необходимо ввести матрицу NxN
print('Осуществляется ввод свободных членов: ')
vector_c = matrix(random=1) # Необходимо ввести матрицу Nx1

matrix_added = []
for j in range(len(vector_c)):
    matrix_added.append(matrix_сoef[j]+vector_c[j])
input_Jacobi = method_Jacobi(matrix_сoef,vector_c)
print('Введенная матрица: ')
print('\n')
print(*matrix_added, sep='\n')
print('\n')
print('Обратная матрица: ')
print('\n')
_ = reverse_matrix(matrix_сoef)
print(*_, sep='\n')
print('\n')
print('Решение методом Якоби: ')
print('\n')
print(input_Jacobi[2])
print('Обусловленность введеной системы:')
print('\n')
print(cord(matrix_сoef))
if cord(matrix_сoef) > 10:
    print("Алгоритм Гаусса")
