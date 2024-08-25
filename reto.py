import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')

# Eliminar columnas innecesarias
df = df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)

# Preprocesamiento de datos
columns_to_convert = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 
    'T', 'RH', 'AH'
]

# Reemplazar comas por puntos y convertir a float64
for column in columns_to_convert:
    if df[column].dtype == 'object':
        df[column] = df[column].str.replace(',', '.')
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Reemplazar los valores -200 por NaN
df[columns_to_convert] = df[columns_to_convert].replace(-200, np.nan)

# Eliminación de filas con valores NaN
df.dropna(subset=columns_to_convert, inplace=True)

# Selección de variables predictoras y objetivo
X = df[['T', 'RH', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)']].values
y = df['C6H6(GT)'].values

X = np.column_stack([np.ones(X.shape[0]), X])

XtX = np.dot(X.T, X)
XtX_inv = np.linalg.inv(XtX)
Xty = np.dot(X.T, y)
beta = np.dot(XtX_inv, Xty)

y_pred = np.dot(X, beta)

mse = np.mean((y - y_pred) ** 2)
r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

print(f'Coeficientes: {beta}')
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Gráfico de dispersión de las predicciones vs valores reales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Valores Predichos vs Valores Reales')
plt.show()

# Distribución de errores
errors = y - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.title('Distribución de los Errores')
plt.xlabel('Error')
plt.show()

# 3. Pairplot para ver relaciones entre variables
plt.figure(figsize=(14, 10))
sns.pairplot(df[['C6H6(GT)', 'T', 'RH', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)']])
plt.suptitle('Relaciones entre las Variables y la Variable Objetivo', y=1.02)
plt.show()

print(df)

print(df.columns)