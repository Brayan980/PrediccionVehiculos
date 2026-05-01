#  Predicción de Valor de Vehículos

**Ciencias de la Computación - CECAR**  
Docente: Carlos Darys Arroyo Pérez

## Descripción

Aplicación web desarrollada con **Streamlit** que implementa tres modelos de Machine Learning para predecir el valor de un vehículo según su año, kilometraje y estado.

## Modelos implementados

| Modelo | Estandarización | R² |
|---|---|---|
| Regresión Lineal | ✅ StandardScaler | ~0.978 |
| KNN Regresión (k=3) | ✅ StandardScaler | ~0.972 |
| Árbol de Decisión | ❌ No requerida | ~0.944 |

## Variables

- **modelo**: Año de fabricación del vehículo
- **kilometraje**: Kilómetros recorridos
- **estado**: Estado del vehículo (escala 1-10)
- **valor_vehiculo**: Valor en dólares (variable objetivo)

## Archivos

```
├── app.py                  # Aplicación Streamlit (interfaz)
├── entrenamiento.py        # Entrenamiento de los 3 modelos
├── prueba.py               # Evaluación y prueba de modelos
├── Valor_vehiculos.csv     # Dataset
├── requirements.txt        # Dependencias
└── README.md
```

## Cómo ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

Aplicación desplegada en **Streamlit Community Cloud**.
