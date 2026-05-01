# ============================================================
# PRUEBA DE MODELOS - Predicción de Valor de Vehículos
# Ciencias de la Computación - CECAR
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

# ============================================================
# 1. Cargar datos y reproducir la misma división
# ============================================================
dataset = pd.read_csv("Valor_vehiculos.csv")

X = dataset[["modelo", "kilometraje", "estado"]]
y = dataset["valor_vehiculo"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 2. Cargar modelos y scaler entrenados
# ============================================================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("modelo_regresion_lineal.pkl", "rb") as f:
    modelo_rl = pickle.load(f)

with open("modelo_knn.pkl", "rb") as f:
    modelo_knn = pickle.load(f)

with open("modelo_arbol.pkl", "rb") as f:
    modelo_arbol = pickle.load(f)

# ============================================================
# 3. Transformar datos de prueba
# ============================================================
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. Predicciones sobre conjunto de PRUEBA
# ============================================================
pred_rl    = modelo_rl.predict(X_test_scaled)
pred_knn   = modelo_knn.predict(X_test_scaled)
pred_arbol = modelo_arbol.predict(X_test)

# ============================================================
# 5. Evaluación de cada modelo
# ============================================================
def evaluar(nombre, y_real, y_pred):
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)
    print(f"\n{'='*45}")
    print(f"  {nombre}")
    print(f"{'='*45}")
    print(f"  MAE  (Error Absoluto Medio) : ${mae:>10,.2f}")
    print(f"  RMSE (Raíz Error Cuadrático): ${rmse:>10,.2f}")
    print(f"  R²   (Coef. Determinación)  : {r2:>11.4f}")
    return mae, rmse, r2

print("\n🔍 EVALUACIÓN DE MODELOS - CONJUNTO DE PRUEBA")

mae_rl,    rmse_rl,    r2_rl    = evaluar("REGRESIÓN LINEAL",    y_test, pred_rl)
mae_knn,   rmse_knn,   r2_knn   = evaluar("KNN REGRESIÓN (k=3)", y_test, pred_knn)
mae_arbol, rmse_arbol, r2_arbol = evaluar("ÁRBOL DE DECISIÓN",   y_test, pred_arbol)

# ============================================================
# 6. Tabla comparativa de predicciones vs valores reales
# ============================================================
print("\n\n📊 COMPARACIÓN PREDICCIONES vs VALORES REALES")
print("="*75)
resultados = pd.DataFrame({
    "Año":         X_test["modelo"].values,
    "Km":          X_test["kilometraje"].values,
    "Estado":      X_test["estado"].values,
    "Real ($)":    y_test.values,
    "Reg. Lineal": pred_rl.round(0).astype(int),
    "KNN":         pred_knn.round(0).astype(int),
    "Árbol":       pred_arbol.round(0).astype(int),
})
print(resultados.to_string(index=False))

# ============================================================
# 7. Predicción con un dato nuevo ingresado manualmente
# ============================================================
print("\n\n🚗 PREDICCIÓN CON DATO NUEVO")
print("-"*45)

# Ingresar por teclado
año_vehiculo = int(input("Ingrese el año del vehículo (ej: 2018): "))
kilometraje  = int(input("Ingrese el kilometraje (ej: 50000): "))
estado       = int(input("Ingrese el estado del vehículo (1-10): "))

# Convertir a matriz (DataFrame)
nuevo_dato = pd.DataFrame([[año_vehiculo, kilometraje, estado]],
                          columns=["modelo", "kilometraje", "estado"])

# Estandarizar para RL y KNN
nuevo_dato_scaled = scaler.transform(nuevo_dato)

# Predicciones
p_rl    = modelo_rl.predict(nuevo_dato_scaled)[0]
p_knn   = modelo_knn.predict(nuevo_dato_scaled)[0]
p_arbol = modelo_arbol.predict(nuevo_dato)[0]

print(f"\n  Vehículo: Año {año_vehiculo} | {kilometraje:,} km | Estado {estado}/10")
print(f"  {'Regresión Lineal':<22}: ${p_rl:>10,.2f}")
print(f"  {'KNN (k=3)':<22}: ${p_knn:>10,.2f}")
print(f"  {'Árbol de Decisión':<22}: ${p_arbol:>10,.2f}")
