# Proyecto ML: predicción de goles del equipo local (Premier League)

Proyecto final de Aprendizaje de Máquina. Partiendo de estadísticas de partidos de fútbol, se evalúan modelos de **clasificación** (entregas previas) y **regresión** (entrega final) para predecir los goles que anota el equipo local (`home_goals`).

**Autores:** Carlos Daniel Martínez Ramírez, Alejandro Ramírez Mejia, Mateo Amaya Moreno.

## Contenido

```
├── notebooks/Proyecto_aprendizaje_premier.ipynb   # Análisis completo (EDA, clasificación, regresión)
├── presentacion/Diapos_proyecto_final_ml.pptx     # Diapositivas de la entrega final
├── requirements.txt
└── README.md
```

## Metodología

1. **EDA:** distribución de goles (sesgo positivo), heatmap de correlación con el Top 20 de variables frente a la victoria local.
2. **Ingeniería de variables:** rachas con `shift(1)` + ventana móvil de 5 partidos (evita *data leakage*), cruce con estadísticas de la temporada previa, imputación de nulos por promedio de equipo y filtro de multicolinealidad (> 0.85), quedando 13 predictoras.
3. **Clasificación** (binaria y multiclase 0 / 1-2 / 3+ goles): Gaussian NB, Decision Tree, SVM y Random Forest, comparados por F1.
4. **Regresión:** Regresión Lineal, Decision Tree, Random Forest, Gradient Boosting y MLP, con `GridSearchCV` (KFold de 5 particiones) y `random_state=18`. Métrica de decisión: RMSE.

## Resultados principales

| Modelo | RMSE (validación cruzada) |
|---|---|
| Random Forest | 1.252 |
| Gradient Boosting | 1.255 |
| Regresión Lineal | 1.259 |

- Baseline (media constante): RMSE test = 1.358.
- Los modelos lo superan, pero con R² ≈ 0.13: el fútbol es muy aleatorio.
- Variables más importantes: diferencia de tiros a puerta, diferencia de goles y pases totales del local.

## Cómo ejecutarlo

```bash
git clone <url-del-repo>
cd proyecto-ml-premier-league
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/Proyecto_aprendizaje_premier.ipynb
```

> Los datos (`stats_df` y `results_df`) se leen desde enlaces de Google Drive dentro de las primeras celdas de código del notebook, así que necesitas conexión a internet y que los enlaces sigan públicos.
