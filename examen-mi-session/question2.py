import numpy as np
import matplotlib.pyplot as plt

f = lambda x : np.sin(x)

x = np.linspace(0, 2*np.pi, 1000)
y = f(x)


dydx = np.gradient(y, x)

maximum = np.empty(0)
minimum = np.empty(0)

for i in range(dydx.shape[0]):
    # Edge cases
    if i == 0:
        if dydx[i] > 0:
            minimum = np.append(minimum, i)
        else:
            maximum = np.append(maximum, i)
    elif i == dydx.shape[0]-1:
        if dydx[i] > 0:
            maximum = np.append(maximum, i)
        else:
            minimum = np.append(minimum, i)
    #General case
    elif dydx[i] > 0 and dydx[i+1] < 0:
        maximum = np.append(maximum, i)
    elif dydx[i] < 0 and dydx[i+1] > 0:
        minimum = np.append(minimum, i)

# Le calcul des maximums et minimums est un peu inutil dans cette situation.
# J'aurai pu directement calculer les extremums.

extremum = np.sort(np.concatenate((minimum, maximum)))

for i in range(extremum.shape[0]):
    if i == extremum.shape[0]-1:
        break
    elif dydx[int((extremum[i] + extremum[i+1])/2)] > 0:
        print("La fonction est croissante lorsque x est élément de", x[int(extremum[i])], "à", x[int(extremum[i+1])])
    else:
        print("La fonction est decroissante lorsque x est élément de", x[int(extremum[i])], "à", x[int(extremum[i+1])])

plt.plot(x,y, "b")
plt.plot(x, dydx, "r")
plt.show()

