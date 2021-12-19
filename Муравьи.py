# Оптимизация колонии Ant для решения проблемы TSP
import numpy as np
import random as rd
from matrix import matrix
import itertools
import networkx as nx
import numpy.random as rnd
import matplotlib.pyplot as plt


def lengthCal(antPath, distmat):  # Рассчитать расстояние
    length = []
    dis = 0
    for i in range(len(antPath)):
        for j in range(len(antPath[i]) - 1):
            dis += distmat[antPath[i][j]][antPath[i][j + 1]]
        dis += distmat[antPath[i][-1]][antPath[i][0]]
        length.append(dis)
        dis = 0
    return length


def add_edge(f_item, s_item, graph=None):
    graph.add_edge(f_item, s_item)
    graph.add_edge(s_item, f_item)


distmat = matrix(square=True)

alpha = 0.6  # Фактор важности феромона
beta = 0.65  # Фактор важности эвристической функции
pheEvaRate = 0.3  # Скорость испарения феромона
cityNum = len(distmat)  # Число городов
antNum = cityNum  # Число муравьев
pheromone = np.ones((cityNum, cityNum))  # Феромоновая матрица
heuristic = 1 / (np.eye(cityNum) + distmat) - np.eye(cityNum)
iter, itermax = 1, 100  # Итерации
antPath = np.zeros((antNum, cityNum)).astype(int) - 1
while iter < itermax:
    antPath = np.zeros((antNum, cityNum)).astype(int) - 1  # Путь муравья
    firstCity = [i for i in range(cityNum)]
    rd.shuffle(firstCity)  # Случайно назначьте начальный город для каждого муравья
    unvisted = []
    p = []
    pAccum = 0
    for i in range(len(antPath)):
        antPath[i][0] = firstCity[i]
    for i in range(len(antPath[0]) - 1):  # Постепенно обновляйте следующий город, в который собирается каждый муравей
        for j in range(len(antPath)):
            for k in range(cityNum):
                if k not in antPath[j]:
                    unvisted.append(k)
            for m in unvisted:
                pAccum += pheromone[antPath[j][i]][m] ** alpha * heuristic[antPath[j][i]][m] ** beta
            for n in unvisted:
                p.append(pheromone[antPath[j][i]][n] ** alpha * heuristic[antPath[j][i]][n] ** beta / pAccum)
            roulette = np.array(p).cumsum()  # Создать рулетку
            r = rd.uniform(min(roulette), max(roulette))
            for x in range(len(roulette)):
                if roulette[x] >= r:  # Используйте метод рулетки, чтобы выбрать следующий город
                    antPath[j][i + 1] = unvisted[x]
                    break
            unvisted = []
            p = []
            pAccum = 0
    pheromone = (1 - pheEvaRate) * pheromone  # Феромон летучий
    length = lengthCal(antPath, distmat)
    for i in range(len(antPath)):
        for j in range(len(antPath[i]) - 1):
            pheromone[antPath[i][j]][antPath[i][j + 1]] += 1 / length[i]  # Обновление феромона
        pheromone[antPath[i][-1]][antPath[i][0]] += 1 / length[i]

    # Визуализация
    graph = nx.Graph()
    for i in range(len(antPath[length.index(min(length))])-1):
        add_edge(antPath[length.index(min(length))][i], antPath[length.index(min(length))][i+1], graph=graph)
    nx.draw_circular(graph,
                     node_color='red',
                     node_size=1000,
                     with_labels=True)
    plt.title(f"Итерация {iter}")
    plt.show()
    iter += 1
print("«Кратчайшее расстояние:»")
print(min(length))
print("«Самый короткий путь:»")
print(antPath[length.index(min(length))])
