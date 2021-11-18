from sympy import symbols, expand, lambdify

def lagranz(x, y):
    t = symbols('t')
    z = 0
    for j in range(len(y)):
        p1 = 1;
        p2 = 1;
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1;
                p2 = p2 * 1
            else:
                p1 = expand(p1 * (t - x[i]))
                p2 = p2 * (x[j] - x[i])
        z = expand(z + y[j] * p1 / p2)
    f_x = lambdify(t, z)
    output = []
    for k in range(len(x)):
        output.append([x[k],y[k],f_x(x[k])])
    return (output, z)
   
   
