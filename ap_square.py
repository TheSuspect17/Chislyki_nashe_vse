from matrix import *
def method_min_square(x,y):
    x_1 = 0
    x_2 = 0
    x_3 = 0
    x_4 = 0
    x2_y = 0
    x_y = 0
    y_1 = 0
    for i in range(len(x)):
        x_1 += x[i]
        x_2 += x[i]**2
        x_3 += x[i]**3
        x_4 += x[i]**4
        x2_y += y[i]*x[i]**2
        x_y += y[i]*x[i]
        y_1 += y[i]
    n = len(x)
    a = [
        [x_2,x_3,x_4],
        [x_1,x_2,x_3],
        [n,x_1,x_2]
        ]
    b = [
        [x2_y],
        [x_y],
        [y_1]
        ]
    roots = gssjrdn(a,b)[2]
    c = []
    for i in range(3):
        c.append(*roots[i])
    def f_x(t):
        return c[2]*t**2+c[1]*t + c[0]
    gamma = 0
    f = []
    for i in range(len(x)):
        f.append(f_x(x[i]))
        gamma += (y[i]-f[i])**2
    exp = ''
    for i in [2,1,0]:
        if c[i] != 0:
            exp += f'{c[i]}*t**{i} + '
    exp = exp[:-2]
    output = [[x[i]]+[y[i]]+[f[i]] for i in range(len(x))]
    return (output,exp, gamma)