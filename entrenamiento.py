# ============================================================
# ENTRENAMIENTO DE MODELOS - Predicción de Valor de Vehículos
# Ciencias de la Computación - CECAR
# ============================================================

# 1. Importar librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ============================================================
# 2. Cargar datos
# ============================================================
dataset = pd.read_csv("Valor_vehiculos.csv")
print("=== DATASET CARGADO ===")
print(dataset.head())
print(f"\nForma del dataset: {dataset.shape}")

# ============================================================
# 3. Separar variables independientes (X) y dependiente (y)
# ============================================================
X = dataset[["modelo", "kilometraje", "estado"]]
y = dataset["valor_vehiculo"]

# ============================================================
# 4. Dividir en conjunto de entrenamiento y prueba (80/20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nEntrenamiento: {X_train.shape[0]} registros")
print(f"Prueba: {X_test.shape[0]} registros")

# ============================================================
# 5. Estandarización (para Regresión Lineal y KNN)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Guardar el scaler para usarlo en la app
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("\n✅ Scaler guardado: scaler.pkl")

# ============================================================
# 6. MODELO 1 - Regresión Lineal (con estandarización)
# ============================================================
print("\n=== MODELO 1: REGRESIÓN LINEAL ===")
modelo_rl = LinearRegression()
modelo_rl.fit(X_train_scaled, y_train)

predicciones_rl = modelo_rl.predict(X_test_scaled)

mae_rl  = mean_absolute_error(y_test, predicciones_rl)
rmse_rl = np.sqrt(mean_squared_error(y_test, predicciones_rl))
r2_rl   = r2_score(y_test, predicciones_rl)

print(f"  MAE  : ${mae_rl:,.2f}")
print(f"  RMSE : ${rmse_rl:,.2f}")
print(f"  R²   : {r2_rl:.4f}")

with open("modelo_regresion_lineal.pkl", "wb") as f:
    pickle.dump(modelo_rl, f)
print("✅ Modelo guardado: modelo_regresion_lineal.pkl")

# ============================================================
# 7. MODELO 2 - KNN Regresión (con estandarización)
# ============================================================
print("\n=== MODELO 2: KNN REGRESIÓN ===")
modelo_knn = KNeighborsRegressor(n_neighbors=3)
modelo_knn.fit(X_train_scaled, y_train)

predicciones_knn = modelo_knn.predict(X_test_scaled)

mae_knn  = mean_absolute_error(y_test, predicciones_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, predicciones_knn))
r2_knn   = r2_score(y_test, predicciones_knn)

print(f"  MAE  : ${mae_knn:,.2f}")
print(f"  RMSE : ${rmse_knn:,.2f}")
print(f"  R²   : {r2_knn:.4f}")

with open("modelo_knn.pkl", "wb") as f:
    pickle.dump(modelo_knn, f)
print("✅ Modelo guardado: modelo_knn.pkl")

# ============================================================
# 8. MODELO 3 - Árbol de Decisión (sin estandarización)
# ============================================================
print("\n=== MODELO 3: ÁRBOL DE DECISIÓN ===")
modelo_arbol = DecisionTreeRegressor(max_depth=4, random_state=42)
modelo_arbol.fit(X_train, y_train)

predicciones_arbol = modelo_arbol.predict(X_test)

mae_arbol  = mean_absolute_error(y_test, predicciones_arbol)
rmse_arbol = np.sqrt(mean_squared_error(y_test, predicciones_arbol))
r2_arbol   = r2_score(y_test, predicciones_arbol)

print(f"  MAE  : ${mae_arbol:,.2f}")
print(f"  RMSE : ${rmse_arbol:,.2f}")
print(f"  R²   : {r2_arbol:.4f}")

with open("modelo_arbol.pkl", "wb") as f:
    pickle.dump(modelo_arbol, f)
print("✅ Modelo guardado: modelo_arbol.pkl")

# ============================================================
# 9. Resumen comparativo
# ============================================================
print("\n" + "="*55)
print("RESUMEN COMPARATIVO DE MODELOS")
print("="*55)
print(f"{'Modelo':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
print("-"*55)
print(f"{'Regresión Lineal':<25} ${mae_rl:>9,.0f} ${rmse_rl:>9,.0f} {r2_rl:>8.4f}")
print(f"{'KNN (k=3)':<25} ${mae_knn:>9,.0f} ${rmse_knn:>9,.0f} {r2_knn:>8.4f}")
print(f"{'Árbol de Decisión':<25} ${mae_arbol:>9,.0f} ${rmse_arbol:>9,.0f} {r2_arbol:>8.4f}")
print("="*55)
