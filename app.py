# ============================================================
# APP STREAMLIT - Predicción de Valor de Vehículos
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Valor de Vehículos", page_icon=None, layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .block-container { padding-top: 2rem; }
        .titulo-principal {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            border-left: 6px solid #4361ee;
            padding-left: 14px;
            margin-bottom: 0.3rem;
        }
        .subtitulo {
            font-size: 1rem;
            color: #aaaaaa;
            margin-bottom: 1.5rem;
            padding-left: 20px;
        }
        .resultado-box {
            background: #1a1a2e;
            color: white;
            border-radius: 10px;
            padding: 1.2rem 2rem;
            font-size: 1.4rem;
            font-weight: 600;
            text-align: center;
            margin-top: 1rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.2rem;
            color: #4361ee;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def entrenar_modelos():
    dataset = pd.read_csv("Valor_vehiculos.csv")
    X = dataset[["modelo", "kilometraje", "estado"]]
    y = dataset["valor_vehiculo"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    modelo_rl = LinearRegression()
    modelo_rl.fit(X_train_scaled, y_train)

    modelo_knn = KNeighborsRegressor(n_neighbors=3)
    modelo_knn.fit(X_train_scaled, y_train)

    modelo_arbol = DecisionTreeRegressor(max_depth=4, random_state=42)
    modelo_arbol.fit(X_train, y_train)

    metricas = {}

    pred_rl = modelo_rl.predict(X_test_scaled)
    metricas["Regresion Lineal"] = {
        "MAE":  mean_absolute_error(y_test, pred_rl),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_rl)),
        "R2":   r2_score(y_test, pred_rl),
    }

    pred_knn = modelo_knn.predict(X_test_scaled)
    metricas["KNN Regresion"] = {
        "MAE":  mean_absolute_error(y_test, pred_knn),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_knn)),
        "R2":   r2_score(y_test, pred_knn),
    }

    pred_arbol = modelo_arbol.predict(X_test)
    metricas["Arbol de Decision"] = {
        "MAE":  mean_absolute_error(y_test, pred_arbol),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_arbol)),
        "R2":   r2_score(y_test, pred_arbol),
    }

    return modelo_rl, modelo_knn, modelo_arbol, scaler, metricas, dataset

modelo_rl, modelo_knn, modelo_arbol, scaler, metricas, dataset = entrenar_modelos()

# Conversión USD a COP
USD_A_COP = 4100

def cop(valor_usd):
    return f"$ {valor_usd * USD_A_COP:,.0f} COP"

# ---- Sidebar ----
st.sidebar.markdown("## Modelo de prediccion")
st.sidebar.markdown("Selecciona el algoritmo:")
modelo_elegido = st.sidebar.radio("", ["Regresion Lineal", "KNN Regresion", "Arbol de Decision"])

# ---- Encabezado ----
st.markdown('<div class="titulo-principal">Prediccion de Valor de Vehiculos</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Ingresa las caracteristicas del vehiculo para estimar su precio de mercado.</div>', unsafe_allow_html=True)

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["Predecir", "Evaluacion de Modelos", "Dataset"])

with tab1:
    st.markdown(f"#### {modelo_elegido}")
    col1, col2, col3 = st.columns(3)
    with col1:
        año = st.number_input("Año del vehiculo", min_value=2000, max_value=2025, value=2018, step=1)
    with col2:
        km = st.number_input("Kilometraje", min_value=0, max_value=300000, value=60000, step=1000)
    with col3:
        estado = st.number_input("Estado (1 = malo, 10 = excelente)", min_value=1, max_value=10, value=8, step=1)

    if st.button("Calcular valor estimado", use_container_width=True):
        nuevo_dato = pd.DataFrame([[año, km, estado]], columns=["modelo", "kilometraje", "estado"])

        if modelo_elegido == "Regresion Lineal":
            pred = modelo_rl.predict(scaler.transform(nuevo_dato))[0]
        elif modelo_elegido == "KNN Regresion":
            pred = modelo_knn.predict(scaler.transform(nuevo_dato))[0]
        else:
            pred = modelo_arbol.predict(nuevo_dato)[0]

        st.markdown(f'<div class="resultado-box">Valor estimado: {cop(pred)}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Comparacion entre los tres modelos:**")
        dato_scaled = scaler.transform(nuevo_dato)
        c1, c2, c3 = st.columns(3)
        c1.metric("Regresion Lineal",  cop(modelo_rl.predict(dato_scaled)[0]))
        c2.metric("KNN Regresion",     cop(modelo_knn.predict(dato_scaled)[0]))
        c3.metric("Arbol de Decision", cop(modelo_arbol.predict(nuevo_dato)[0]))

with tab2:
    st.markdown("#### Resultados sobre el conjunto de prueba (20% de los datos)")
    st.markdown("Mide que tan bien predice cada modelo con datos que no vio durante el entrenamiento.")
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    for col, nombre in zip([c1, c2, c3], ["Regresion Lineal", "KNN Regresion", "Arbol de Decision"]):
        m = metricas[nombre]
        with col:
            st.markdown(f"**{nombre}**")
            st.metric("R2", f"{m['R2']:.4f}")
            st.metric("MAE", cop(m['MAE']))
            st.metric("RMSE", cop(m['RMSE']))

    st.markdown("---")
    st.info(
        "R2 cercano a 1 indica mejor ajuste.\n\n"
        "MAE y RMSE en COP: entre menor, mejor.\n\n"
        "Regresion Lineal y KNN usan estandarizacion (StandardScaler). "
        "Arbol de Decision no la requiere."
    )

with tab3:
    st.markdown(f"#### Dataset — {len(dataset)} registros, {dataset.shape[1]} variables")
    st.dataframe(dataset, use_container_width=True)
    st.markdown("**Estadisticas descriptivas:**")
    st.dataframe(dataset.describe().round(2), use_container_width=True)
