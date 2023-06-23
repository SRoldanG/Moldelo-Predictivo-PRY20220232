import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA


archivo_csv = "C:/Users/alexa/PycharmProjects/pythonProject/Borrador/DatasetdePrueba4.csv"
data = pd.read_csv(archivo_csv, delimiter=";")

le = LabelEncoder()
data['Género'] = le.fit_transform(data['Género'])
print(data)

X = data.drop(['Nombre','Carrera Elegida','Modalidad de Clases','Cursos Aprobados','Desempeño','Desertará'], axis=1)
y = data['Desertará']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression()
clf.fit(X_scaled, y)

predictions = clf.predict(X_scaled)
probabilities = clf.predict_proba(X_scaled)

data['prediccion_desercion'] = predictions
data['probabilidad_desercion'] = probabilities[:, 1]* 100



def calcular_riesgo_desercion(probabilidad_desercion):
    if probabilidad_desercion < 50:
        return 'Bajo'
    elif 50 <= probabilidad_desercion < 75:
        return 'Medio'
    else:
        return 'Alto'

data['riesgo_desercion'] = data['probabilidad_desercion'].apply(calcular_riesgo_desercion)

print(data)

# Obtener los coeficientes del modelo de Regresión Logística
coeficientes = clf.coef_[0]

# Asociar los coeficientes con sus respectivos nombres de característica
caracteristicas_coeficientes = dict(zip(X.columns, coeficientes))

# Ordenar las características por importancia (valor absoluto del coeficiente)
caracteristicas_ordenadas = sorted(caracteristicas_coeficientes.items(), key=lambda x: abs(x[1]), reverse=True)

# Imprimir las características y sus coeficientes
print("Características y sus coeficientes:")
for caracteristica, coeficiente in caracteristicas_ordenadas:
    print(f"{caracteristica}: {coeficiente:.4f}")

# Primero, estandariza el conjunto de datos original
X_scaled = scaler.transform(X)

# Calcula el impacto de cada característica en el riesgo de deserción
impacto = X_scaled * coeficientes

# Encuentra el factor con el mayor impacto (valor absoluto) para cada alumno
factor_influyente = X.columns[np.abs(impacto).argmax(axis=1)]

# Agrega la columna "factor influyente" al DataFrame
data['factor_influyente'] = factor_influyente

# Muestra el DataFrame actualizado
print(data)

# Evaluar el desempeño del modelo

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrena el modelo en el conjunto de entrenamiento
clf.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcula métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
accuracy = accuracy * 100
recall = recall_score(y_test, y_pred)
recall = recall * 100
f1 = f1_score(y_test, y_pred)
f1 = f1 * 100
confusion = confusion_matrix(y_test, y_pred)

print("\n")
print(f"Accuracy: {accuracy:.2f} %")
print(f"Recall: {recall:.2f} %")
print(f"F1 score: {f1:.2f} %")





# Calcular la matriz de correlación
correlation_matrix = X.corr()

# Imprimir la matriz de correlación
print(correlation_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()


# Calcula la correlación y el p-valor entre 'Edad' y 'Tiene Trabajo'
correlation, p_value = stats.pearsonr(data['Edad'], data['Tiene Trabajo'])

print(f'Correlation: {correlation}')
print(f'p-value: {p_value}')

#print("Confusion Matrix:")
#print(confusion)
#data.to_csv("C:/Users/alexa/PycharmProjects/pythonProject/Borrador/DatasetdePrueba9.csv", index=False, sep=";")

