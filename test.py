def opredelitel(matrix,mult):
    porydok_matrix = len(matrix)
    if porydok_matrix == 1:
        return mult*matrix[0][0]
    else:
        sign = -1
        sum = 0
    for i in range(porydok_matrix): # 0 1
        minor = []
        for j in range(1, porydok_matrix): # 1
            stroka_minora = []
            for k in range(porydok_matrix): # 0 1
                if k != i:
                    stroka_minora.append(matrix[j][k])

            minor.append(stroka_minora)
        sign *= -1
        sum += mult * opredelitel(minor, sign * matrix[0][i])
    return sum
