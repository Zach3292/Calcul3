import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def simpson(f, a, b, n) :

    # INSCRIRE DU CODE ICI
    x = np.linspace(a, b, n+1)
    y = f(x)
    dx = (b - a) / n

    integrale = np.sum(np.fromiter((
    y[2*i] + 4*y[2*i+1] + y[2*i+2] for i in range(int(n/2))
    ), dtype=float)) * dx / 3 # 3 parce que on a n/2 intervalles au lieu de n

    return integrale

# Spécifier la moyenne et l'écart type de la loi normale
mu = float(input("Moyenne de la loi normale : "))
sigma = float(input("Écart type de la loi normale : "))

# Définition de la fonction normale à partir de mu et sigma

# INSCRIRE DU CODE ICI
# Vous devez changer la fonction pour celle décrivant une loi normale de moyenne mu et d'écart type sigma
f = lambda x : norm.pdf(x, loc=mu, scale=sigma) # TODO : voir si c'est la bonne fonction

# Graphique de la courbe associée à la loi normale
x = np.linspace(mu-4 * sigma, mu+4 * sigma, 1000)
y = f(x)
ax = plt.subplot(111)
ax.plot(x, y, label="N(%.2f, %.2f)" % (mu, sigma**2), color='dodgerblue')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Spécifier le type de probabilité à évaluer
print("Quel type de probabilité voulez-vous évaluer ?")
print("\t \t 1. P(X < a)")
print("\t \t 2. P(a < X < b)")
print("\t \t 3. P(X > a)")
type_probabilite = int(input("Type de probabilité : "))
while type_probabilite not in {1, 2, 3}:
    type_probabilite = int(input("Choisir le type 1, 2 ou 3 : "))

erreur_maximale = 0.001

if type_probabilite == 1:
    a = float(input("Valeur du a dans P(X < a) : "))
    borne_inferieure = -np.inf

    # Évaluation de la probabilité
    if a < mu:
        p = 0
        # Calcul du nombre de sous intervalles nécessaire

        # INSCRIRE DU CODE ICI
        n = (( ((a - borne_inferieure) ** 5) / (180 * erreur_maximale) ) * val_max_dev_4) ** (1/4)
        # TODO : demander au prof pour la borne inférieure et changer la valeur maximale de la dérivée 4ieme

        # Évaluation de la probabilité
        
        # INSCRIRE DU CODE ICI
        p = simpson(f, borne_inferieure, a, n)
    else:
        p = 0
        # Calcul du nombre de sous intervalles nécessaire

        # INSCRIRE DU CODE ICI
        n = (( ((a - borne_inferieure) ** 5) / (180 * erreur_maximale) ) * val_max_dev_4) ** (1/4)
        # TODO : demander au prof pour la borne inférieure et changer la valeur maximale de la dérivée 4ieme
        
        # Évaluation de la probabilité
        
        # INSCRIRE DU CODE ICI
        p = simpson(f, borne_inferieure, a, n)

    print("La probabilité P(X < %.3f) vaut %.6f" % (a, p))

    # Tracer la région associée à la probabilité
    x_prob = np.linspace(mu-4 * sigma, a, 100)
    y_prob = f(x_prob)
    plt.vlines(a, ymin=0, ymax=f(a), color='dodgerblue')
    plt.fill_between(x_prob, y_prob, color='skyblue', label="P(X < %.3f) = %.6f" % (a, p))
    plt.title("Loi normale de moyenne %.2f et d\'écart type %.2f" % (mu, sigma))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075))

elif type_probabilite == 2:
    a = float(input("Valeur du a dans P(a < X < b) : "))
    b = float(input("Valeur du b dans P(a < X < b) : "))
    while b < a:
        print("Il faut que a soit inférieur à b.")
        a = float(input("Valeur du a dans P(a < X < b) : "))
        b = float(input("Valeur du b dans P(a < X < b) : "))

    p = 0
    # Calcul du nombre de sous intervalles nécessaire

    # INSCRIRE DU CODE ICI
    n = (( ((b - a) ** 5) / (180 * erreur_maximale) ) * val_max_dev_4) ** (1/4)
    # TODO : changer la valeur maximale de la dérivée 4ieme

    # Évaluation de la probabilité
    
    # INSCRIRE DU CODE ICI
    p = simpson(f, a, b, n)
    print("La probabilité P(%.3f < X < %.3f) vaut %.6f" % (a, b, p))

    # Tracer la région associée à la probabilité
    x_prob = np.linspace(a, b, 100)
    y_prob = f(x_prob)
    plt.vlines(a, ymin=0, ymax=f(a), color='dodgerblue')
    plt.vlines(b, ymin=0, ymax=f(b), color='dodgerblue')
    plt.fill_between(x_prob, y_prob, color='skyblue', label="P(%.3f < X < %.3f) = %.6f" % (a, b, p))
    plt.title("Loi normale de moyenne %.2f et d\'écart type %.2f" % (mu, sigma))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075))

elif type_probabilite == 3:
    a = float(input("Valeur du a dans P(X > a) : "))
    borne_superieur = np.inf
    # Évaluation de la probabilité
    if a < mu:
        p = 0
        # Calcul du nombre de sous intervalles nécessaire

        # INSCRIRE DU CODE ICI
        n = (( ((borne_superieur - a) ** 5) / (180 * erreur_maximale) ) * val_max_dev_4) ** (1/4)
        # TODO : demander au prof pour la borne supérieure et changer la valeur maximale de la dérivée 4ieme

        # Évaluation de la probabilité
        
        # INSCRIRE DU CODE ICI
        p = simpson(f, a, borne_superieur, n)
    else:
        p = 0
        # Calcul du nombre de sous intervalles nécessaire

        # INSCRIRE DU CODE ICI
        n = (( ((borne_superieur - a) ** 5) / (180 * erreur_maximale) ) * val_max_dev_4) ** (1/4)
        # TODO : demander au prof pour la borne supérieure et changer la valeur maximale de la dérivée 4ieme

        # Évaluation de la probabilité
        
        # INSCRIRE DU CODE ICI
        p = simpson(f, a, borne_superieur, n)

    print("La probabilité P(X > %.3f) vaut %.6f" % (a, p))

    # Tracer la région associée à la probabilité
    x_prob = np.linspace(a, mu+4*sigma, 100)
    y_prob = f(x_prob)
    plt.vlines(a, ymin=0, ymax=f(a), color='dodgerblue')
    plt.fill_between(x_prob, y_prob, color='skyblue', label="P(X > %.3f) = %.6f" % (a, p))
    plt.title("Loi normale de moyenne %.2f et d\'écart type %.2f" % (mu, sigma))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075))

plt.show()
