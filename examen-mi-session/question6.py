import numpy as np
import scipy.integrate as scpint

f = lambda x: np.sin(x)**2 + x**2
a = 0
b = np.pi 
n = 10

x = np.linspace(a, b, n + 1)
y = f(x)

fonctionExacte = lambda x: (1/6)*(2*x**3+3*x-3*np.sin(x)*np.cos(x))

integrale = scpint.trapezoid(y, x)
print("Intégrale avec", n, "trapèzes:", integrale)
erreur = np.abs((fonctionExacte(b)-fonctionExacte(a))-integrale)
print("Erreur avec", n, "trapèzes:", erreur)
integrale = scpint.simpson(y, x)
print("Intégrale avec", n, "sous-intervalles et simpson:", integrale)
erreur = np.abs((fonctionExacte(b)-fonctionExacte(a))-integrale)
print("Erreur avec", n, "sous-intervalles et simpson:", erreur)