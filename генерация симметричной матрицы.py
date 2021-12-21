def generante_matrix(n):
    a = []
    for i in range(n):
        t = []
        for j in range(n):
            t.append(0)
        a.append(t)
    for i in range(n):
        for j in range(i,n):
            if i == j:
                a[i][j] = 0
            else:
                _ = randint(10,30)
                a[i][j] = _
                a[j][i] = _
    return a

