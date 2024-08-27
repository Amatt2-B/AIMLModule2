import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')

# Preprocesamiento de datos
df = df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)
columns_to_convert = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 
                      'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 
                      'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

for column in columns_to_convert:
    if df[column].dtype == 'object':
        df[column] = df[column].str.replace(',', '.')
    df[column] = pd.to_numeric(df[column], errors='coerce')

df[columns_to_convert] = df[columns_to_convert].replace(-200, np.nan)
df.dropna(subset=columns_to_convert, inplace=True)

# Selección de variables predictoras y objetivo
X = df[['T', 'RH', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)']].values
y = df['C6H6(GT)'].values

# Normalización manual de las variables predictoras
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Añadir la columna de 1s para el intercepto
X = np.column_stack([np.ones(X.shape[0]), X])

# Inicializar los parámetros
params = np.zeros(X.shape[1])

# Hiperparámetros
learning_rate = 0.001
epochs = 2000

# Implementar Descenso de Gradiente
m = len(y)
for epoch in range(epochs):
    predictions = np.dot(X, params)
    errors = predictions - y
    gradients = (2/m) * np.dot(X.T, errors)
    params -= learning_rate * gradients
    
    if epoch % 100 == 0:
        mse = np.mean(errors**2)
        print(f'Epoch {epoch}: MSE = {mse}')

# Resultados
print(f'Parámetros finales: {params}')

# Predicciones
y_pred = np.dot(X, params)

# Evaluación del Modelo
mse = np.mean((y - y_pred) ** 2)
r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Gráficos
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Valores Predichos vs Valores Reales')
plt.show()
