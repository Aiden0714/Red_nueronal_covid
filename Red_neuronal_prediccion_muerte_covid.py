
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier

nom_archivo="dataset_covid_noheaders.csv"
datos=np.loadtxt(nom_archivo, delimiter=",")
print(datos.shape) #Imprimir dimensiones del conjunto de datos

X=datos[:,0:24] #Caracteristicas del conjunto de datos
y=datos[:,24] #Etiquetas (clases)

print("  ▒ ▒ ▒ ▒ ▒ ▒ ▒   Primeras lineas CSV Original­­  ▒ ▒ ▒ ▒ ▒ ▒ ▒\n")
print(X[:10]) #Imprime las primera cinco lineas del data set
print(y[:10]) #Imprime las primeras cinco etiquetas


#Conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

print(X_train.shape) #Tamaño del conjunto de entrenamiento
print(X_test.shape) #Tamaño del conjunto de prueba

#Estandarizacion de los conjuntos de entrenamiento y prueba
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test) 

print("\n▒ ▒ ▒ ▒ ▒ ▒ ▒   ESTANDARIZACION  ▒ ▒ ▒ ▒ ▒ ▒ ▒\n")
print(" ===Entrenamiento===\n")
print(X_train[:5])
print(y_train[:5])
print(" \n===Prueba===\n")
print(X_test[:5])
print(y_test[:5])


clf=MLPClassifier(random_state=42,max_iter=1000)
clf.fit(X_train, y_train)

accuracy=round(clf.score(X_test, y_test),3)
print("***Exactitud: ",accuracy,"***")

