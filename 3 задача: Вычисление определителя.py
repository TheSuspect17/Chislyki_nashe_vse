def determinant(matr):
    size = len(matr)
    if size == 1:
        return matr[0][0]
    for k in range(size):
        t = [ row[:k] + row[k+1:] for row in (matr[1:])]
        matr[0][k] = matr[0][k] * (-1)**(k) * determinant(t)
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
matr = []
for i in range(m):
    t = []
    for j in range(m):
        _ = input(f'Введите элемент {i + 1} строки {j + 1} столбца: ')
        #_ = rd.randint(0,100)
        #_ = 0
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

for i in range(len(matr)):
    print(matr[i])

print(determinant(matr))
