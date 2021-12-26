
import numpy as np
import pywt
from sympy import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv

with open("d://Test_sin.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    qwe = []
    for row in reader:
        qwe.append(list(row))
for i in range(1, len(qwe)):
    qwe[i][0] = float(qwe[i][0])
    qwe[i][1] = float(qwe[i][1])


def wavelet_sinus(coeffs):
    a4, d4, d3, d2, d1 = coeffs
    result = np.sqrt(np.std(d4) + np.std(d3) + np.std(d2) + np.std(d1)) / np.std(a4)
    return result


x = [i[0] for i in qwe]
y = [i[1] for i in qwe]
coeffs = pywt.wavedec(y, 'db1', level=4)

# Уровень 1

rcParams['figure.figsize'] = (10, 10)
rcParams['figure.dpi'] = 100

plt.plot(coeffs[-4][int(len(coeffs[-4])*0.05):int(len(coeffs[-4])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Добеши Уровень 1. ', fontsize=24, loc='center')
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()

# Уровень 2
coeffs = pywt.wavedec(y, 'db2', level=4)

plt.plot(coeffs[-3][int(len(coeffs[-3])*0.05):int(len(coeffs[-3])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Добеши Уровень 2. ', fontsize=24, loc='center')
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()

# Уровень 3

coeffs = pywt.wavedec(y, 'db3', level=4)
plt.plot(coeffs[-2][int(len(coeffs[-2])*0.05):int(len(coeffs[-2])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Добеши Уровень 3.', fontsize=24, loc='center')

plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()

# Уровень 4
coeffs = pywt.wavedec(y, 'db4', level=4)

plt.plot(coeffs[-1][int(len(coeffs[-1])*0.05):int(len(coeffs[-1])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Добеши Уровень 4. ', fontsize=24, loc='center')

plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()

coeffs = pywt.wavedec(y, 'db5')

print(f'Коэффициент несинусоидальности: {wavelet_sinus(coeffs[:5])}.')

x = [i[0] for i in qwe]
y = [i[1] for i in qwe]
coeffs = pywt.wavedec(y, 'haar', level=4)

plt.plot(coeffs[-4][int(len(coeffs[-4])*0.05):int(len(coeffs[-4])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Haar Уровень 1.', fontsize=24, loc='center')

plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()

plt.plot(coeffs[-3][int(len(coeffs[-3])*0.05):int(len(coeffs[-3])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Haar Уровень 2. ', fontsize=24, loc='center')

plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()

plt.plot(coeffs[-2][int(len(coeffs[-2])*0.05):int(len(coeffs[-2])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Haar Уровень 3', fontsize=24, loc='center')

plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()

plt.plot(coeffs[-1][int(len(coeffs[-1])*0.05):int(len(coeffs[-1])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Haar Уровень 4.', fontsize=24, loc='center')
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()

coeffs = pywt.wavedec(y, 'haar')
print(f'Коэффициент несинусоидальности: {wavelet_sinus(coeffs[:5])}.')

x = [i[0] for i in qwe]
y = [i[1] for i in qwe]
wavelet = pywt.ContinuousWavelet('mexh')
coef, freqs = pywt.cwt(y, np.arange(1, 30), wavelet)
endcoef = []
endcoef.append(coef[0])

plt.plot(coef[0][int(len(coef[0])*0.05):int(len(coef[0])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Шляпа Уровень 1', fontsize=24, loc='center')

plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()
coef, freqs = pywt.cwt(coef[0], np.arange(1, 30), wavelet)

plt.plot(coef[0][int(len(coef[0])*0.05):int(len(coef[0])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Шляпа Уровень 2', fontsize=24, loc='center')

plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()
coef, freqs = pywt.cwt(coef[0], np.arange(1, 30), wavelet)
endcoef.append(coef[0])

plt.plot(coef[0][int(len(coef[0])*0.05):int(len(coef[0])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Шляпа Уровень 3.', fontsize=24, loc='center')

plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()
coef, freqs = pywt.cwt(coef[0], np.arange(1, 30), wavelet)
endcoef.append(coef[0])

plt.plot(coef[0][int(len(coef[0])*0.05):int(len(coef[0])*0.95)], color='black', lw=2)
plt.grid(b=True, color='black', alpha=0.75, linestyle=':', linewidth=1)
plt.title(f'Шляпа Уровень 4', fontsize=24, loc='center')

plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.show()
coef, freqs = pywt.cwt(coef[0], np.arange(1, 30), wavelet)
endcoef.append(coef[0])

endcoef.append(np.sqrt(np.var(coef[-1])))
endcoef.reverse()
coefsin = np.sqrt(np.std(endcoef[1]) + np.std(endcoef[2]) + np.std(endcoef[3]) + np.std(endcoef[4])) / endcoef[0]
print(f'Коэффициент несинусоидальности: {coefsin}.')
