# ============================================================
# APP STREAMLIT - Predicción de Valor de Vehículos
# Ciencias de la Computación - CECAR
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Configuración de la página
# ============================================================
st.set_page_config(
    page_title="Predicción Valor Vehículos",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# Entrenar modelos en caché (se ejecuta solo una vez)
# ============================================================
@st.cache_resource
def entrenar_modelos():
    dataset = pd.read_csv("Valor_vehiculos.csv")

    X = dataset[["modelo", "kilometraje", "estado"]]
    y = dataset["valor_vehiculo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Estandarización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Regresión Lineal
    modelo_rl = LinearRegression()
    modelo_rl.fit(X_train_scaled, y_train)

    # KNN
    modelo_knn = KNeighborsRegressor(n_neighbors=3)
    modelo_knn.fit(X_train_scaled, y_train)

    # Árbol de Decisión
    modelo_arbol = DecisionTreeRegressor(max_depth=4, random_state=42)
    modelo_arbol.fit(X_train, y_train)

    # Métricas
    metricas = {}

    pred_rl = modelo_rl.predict(X_test_scaled)
    metricas["Regresión Lineal"] = {
        "MAE":  mean_absolute_error(y_test, pred_rl),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_rl)),
        "R²":   r2_score(y_test, pred_rl),
    }

    pred_knn = modelo_knn.predict(X_test_scaled)
    metricas["KNN Regresión"] = {
        "MAE":  mean_absolute_error(y_test, pred_knn),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_knn)),
        "R²":   r2_score(y_test, pred_knn),
    }

    pred_arbol = modelo_arbol.predict(X_test)
    metricas["Árbol de Decisión"] = {
        "MAE":  mean_absolute_error(y_test, pred_arbol),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_arbol)),
        "R²":   r2_score(y_test, pred_arbol),
    }

    return modelo_rl, modelo_knn, modelo_arbol, scaler, metricas, dataset

modelo_rl, modelo_knn, modelo_arbol, scaler, metricas, dataset = entrenar_modelos()

# ============================================================
# Sidebar - Selección de modelo
# ============================================================
st.sidebar.title("Seleccionar Modelo")
st.sidebar.write("Elige un modelo:")
modelo_elegido = st.sidebar.radio(
    "",
    ["Árbol de Decisión", "KNN Regresión", "Regresión Lineal"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**CECAR**")
st.sidebar.markdown("Ciencias de la Computación")
st.sidebar.markdown("Carlos Darys Arroyo Pérez")

# ============================================================
# Título principal
# ============================================================
st.title("🚗 Modelos de Predicción")
st.markdown("### Predicción del Valor de Vehículos")
st.markdown("---")

# ============================================================
# Tabs: Predicción | Evaluación | Dataset
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predecir", "📊 Evaluación de Modelos", "📋 Dataset"])

# ---- TAB 1: PREDICCIÓN ----
with tab1:
    st.subheader(f"{modelo_elegido} - Predicción de Valor de Vehículo")

    col1, col2, col3 = st.columns(3)

    with col1:
        año = st.number_input("Año del vehículo:", min_value=2000, max_value=2025,
                               value=2018, step=1)
    with col2:
        km = st.number_input("Kilometraje:", min_value=0, max_value=300000,
                              value=60000, step=1000)
    with col3:
        estado = st.number_input("Estado del vehículo (1-10):", min_value=1,
                                  max_value=10, value=8, step=1)

    st.markdown("")

    if st.button("🔍 Predecir valor de vehículo", use_container_width=True):
        nuevo_dato = pd.DataFrame([[año, km, estado]],
                                  columns=["modelo", "kilometraje", "estado"])

        if modelo_elegido == "Regresión Lineal":
            dato_scaled = scaler.transform(nuevo_dato)
            prediccion = modelo_rl.predict(dato_scaled)[0]
        elif modelo_elegido == "KNN Regresión":
            dato_scaled = scaler.transform(nuevo_dato)
            prediccion = modelo_knn.predict(dato_scaled)[0]
        else:  # Árbol de Decisión
            prediccion = modelo_arbol.predict(nuevo_dato)[0]

        st.success(f"💰 **Valor estimado del vehículo: ${prediccion:,.2f} USD**")

        # Mostrar también los otros modelos
        st.markdown("#### Comparación entre todos los modelos:")
        dato_scaled = scaler.transform(nuevo_dato)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Regresión Lineal",  f"${modelo_rl.predict(dato_scaled)[0]:,.0f}")
        col_b.metric("KNN Regresión",     f"${modelo_knn.predict(dato_scaled)[0]:,.0f}")
        col_c.metric("Árbol de Decisión", f"${modelo_arbol.predict(nuevo_dato)[0]:,.0f}")

# ---- TAB 2: EVALUACIÓN ----
with tab2:
    st.subheader("📊 Evaluación de los Modelos (conjunto de prueba)")
    st.markdown("Métricas calculadas sobre el 20% de datos reservados para prueba (6 registros).")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    for col, nombre in zip([col1, col2, col3],
                           ["Regresión Lineal", "KNN Regresión", "Árbol de Decisión"]):
        m = metricas[nombre]
        with col:
            st.markdown(f"**{nombre}**")
            st.metric("R² (Coef. Determinación)", f"{m['R²']:.4f}")
            st.metric("MAE (Error Absoluto Medio)", f"${m['MAE']:,.2f}")
            st.metric("RMSE (Raíz Cuad. Medio)",   f"${m['RMSE']:,.2f}")

    st.markdown("---")
    st.markdown("**Interpretación:**")
    st.info(
        "• **R²** cercano a 1 indica mejor ajuste del modelo.\n"
        "• **MAE** y **RMSE** en dólares: entre menor, mejor.\n"
        "• **Regresión Lineal y KNN** usan estandarización (StandardScaler).\n"
        "• **Árbol de Decisión** no requiere estandarización."
    )

# ---- TAB 3: DATASET ----
with tab3:
    st.subheader("📋 Dataset - Valor de Vehículos")
    st.markdown(f"Total de registros: **{len(dataset)}** | Variables: **{dataset.shape[1]}**")
    st.dataframe(dataset, use_container_width=True)
    st.markdown("**Descripción estadística:**")
    st.dataframe(dataset.describe().round(2), use_container_width=True)
