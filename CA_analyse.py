# Linear regression 
# Author: Timothée Charrier

import numpy as np
import matplotlib.pyplot as plt

X_real = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Y_real = [0.0, 0.0, 2.2495431129078924E-5, 0.1804257821254055, 0.5471663577941505, 0.9235531331127038, 1.303852164794697, 1.686192712473536, 2.069750734467568, 2.454131417920758, 2.8391023014585945, 3.224518726477628]

X_10_20 = [10, 20]
Y_10_20 = [2.2495431129078924E-5, 0.1804257821254055]

X_20_100 = [20, 30, 40, 50, 60, 70, 80, 90, 100]
Y_20_100 = [0.1804257821254055, 0.5471663577941505, 0.9235531331127038, 1.303852164794697, 1.686192712473536, 2.069750734467568, 2.454131417920758, 2.8391023014585945, 3.224518726477628]


def linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    return m, c

m, c = linear_regression(X_20_100, Y_20_100)

m2, c2 = linear_regression(X_10_20, Y_10_20)

print('Linear regression on 10-20% on data between 20-100% is y = {}x + {}'.format(m, c))
print('Linear regression on 10-20% on data between 10-20% is y = {}x + {}'.format(m2, c2))

X = np.linspace(0, 100, 100)
Y1 = m*X + c
Y2 = m2*X + c2

plt.plot(X_real, Y_real, 'o', label='Données originales entre 0-10 et 20-100%', markersize=10)
plt.plot(X, Y1, 'r', label='Regression lineaire sur 10-20% sur les données entre 20-100%')
plt.plot(X_10_20, Y_10_20, 'o', label='Données originales entre 10-20%', markersize=10)
plt.plot(X, Y2, 'g', label='Regression lineaire sur 10-20% sur les données entre 10-20%')
plt.legend()
plt.show()
plt.savefig('CA_analyse.png')

# Error between Y1 and Y2 on X_10_20 in percentage
error = np.abs((Y1[10:20] - Y2[10:20]) / Y1[10:20]) * 100
print(error)
print(np.mean(error))

# Perform polynomial regression on X_real and Y_real
z = np.polyfit(X_real, Y_real, 3)
f = np.poly1d(z)
print(f)

# Plot the polynomial regression

X = np.linspace(0, 100, 100)
Y = f(X)

plt.plot(X_real, Y_real, 'o', label='Données originales entre 0-100%', markersize=10)
plt.plot(X, Y, 'r', label='Regression polynomiale sur les données entre 0-100%')
plt.legend()
plt.show()
plt.savefig('CA_non_linear_analyse.png')