# MINI ML PROJEKAT: Predikcija plate na osnovu iskustva

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Podaci (godine iskustva)
X = np.array([[1], [2], [3], [4], [5]])

# Plate (u nekoj jedinici)
y = np.array([300, 400, 500, 600, 700])

# Kreiranje modela
model = LinearRegression()


# Treniranje modela
model.fit(X, y)

# Predikcija
pred = model.predict(X)

# Ispis
print("Predikcija za 6 godina iskustva:", model.predict([[6]]))

# Graf
plt.scatter(X, y)
plt.plot(X, pred)
plt.xlabel("Godine iskustva")
plt.ylabel("Plata")
plt.title("Linearna regresija")
plt.show()