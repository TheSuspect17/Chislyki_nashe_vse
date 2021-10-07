def determinant(matr):
    #print('Произошла рекурсия')
    size = len(matr)
    if size == 1:
        return matr[0][0]
    for k in range(size):
        #print(f"Работаем с {k} элементом строки")
        t = [ row[:k] + row[k+1:] for row in (matr[1:])]
        #print(t)
        matr[0][k] = matr[0][k] * (-1)**(k) * determinant(t)
    return sum(matr[0])
