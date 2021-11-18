from sympy import symbols, expand, lambdify

def lagranz(x, y):
    t = symbols('t')
    z = 0
    for j in range(len(y)):
        numerator = 1;
        denominator = 1;
        for i in range(len(x)):
            if i == j:
                numerator = numerator * 1;
                denominator = denominator * 1
            else:
                numerator = expand(numerator * (t - x[i]))
                denominator = denominator * (x[j] - x[i])
        z = expand(z + y[j] * numerator / denominator)
    f_x = lambdify(t, z)
    output = []
    for k in range(len(x)):
        output.append([x[k],y[k],f_x(x[k])])
    return (output, z)

   
# Вывод в формате ([[x1,y1,f_x1]...[xi,yi,f_xi]], f_x) 
# f_x - строка, можно преобразовать в функцию типа f_x(x0) = y0  
