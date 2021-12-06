from sympy import symbols, lambdify, sqrt, cos, sin, log, exp
def input_function():
    t = symbols('t')
    y = symbols('y')
    z = symbols('z')
    equation = []

    qwe = eval(input('Введите y` [доступные символы t,y,z]: '))
    equation.append(qwe)
    global answer
    answer = input('Хотите ввести еще одно уравнение? [1/0] ')
    if answer == '1':
        qwe = eval(input('Введите z` [доступные символы t,y,z]: '))
        equation.append(qwe)
    function = []
    for i in range(len(equation)):
        function.append(lambdify([t, y, z], equation[i]))
    return function
 
# Может багаться
