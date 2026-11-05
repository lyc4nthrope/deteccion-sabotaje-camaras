# Clasificadores supervisados sobre los descriptores clásicos (T45)

Clase 20: KNN, SVM y Naive Bayes. Código: `src/ml_clasico.py`. Resultados: `results/clasico/clasificadores.csv`.
Reproducir con `python -m src.ml_clasico` (después de `python -m src.build_features`).

## 1. Qué se hizo y por qué

Las reglas (T40–T43) deciden con umbrales escritos a mano sobre el vector de descriptores. Aquí, sobre **el
mismo vector**, se entrena un clasificador que aprende la decisión de las etiquetas. Es un punto intermedio entre
las reglas y la CNN: misma información de entrada que las reglas, pero decisión aprendida, como la CNN.

| Decisión | Por qué |
|---|---|
| Logaritmo de `lap_var`, `orb_matches` y `orb_inliers`, y luego estandarización | KNN y SVM usan distancias y son sensibles a la escala (clase 20): sin esto, el Laplaciano (0 a más de 2000) dominaría sobre la saturación (0 a 1) |
| Ángulo faltante codificado como `angle_missing` (0/1) y `abs_angle` = 0 | "No hay ángulo confiable" es información; los modelos no aceptan valores vacíos |
| Hiperparámetros con validación cruzada de 5 pliegues **agrupada por bloques de tiempo** (`StratifiedGroupKFold`), solo en train, con F1 macro | Un K-fold aleatorio pondría fotos casi idénticas, tomadas con un minuto de diferencia, en entrenamiento y validación |
| SVM sin probabilidades; para la curva ROC se usa su función de decisión | Las probabilidades de la SVM requieren una calibración interna con pliegues aleatorios, justo la fuga que se evita |
| La cámara no vista **no se usó para elegir nada** | Lección del módulo clásico (`docs/clasico.md`, sección 5) |

Hiperparámetros elegidos: KNN con k = 5, distancia Manhattan y votos ponderados por distancia; SVM lineal con
C = 1; Naive Bayes con `var_smoothing` = 0,001.

## 2. Resultados

| Modelo | F1 (CV en train) | Val F1 | **Test F1** | **Cámara no vista F1** | Falsas alarmas (no vista) | AUC (no vista) |
|---|---|---|---|---|---|---|
| **Reglas (T43)** | — | 0,963 | **0,954** | **0,920** | 0,1 % | 0,979 |
| KNN | 0,942 | 0,948 | 0,888 | 0,451 | 6,8 % | 0,807 |
| SVM | 0,940 | 0,974 | 0,895 | 0,422 | 28,3 % | 0,973 |
| Naive Bayes | 0,863 | 0,852 | 0,919 | 0,339 | 3,8 % | 0,956 |
| CNN (T64, PR #11) | — | — | 0,967 | 0,580 | 4,1 % | 0,752 |

Tiempo del modelo por imagen, sin contar el cálculo de los descriptores (≈ 13 ms, igual que las reglas): KNN
0,9 ms, SVM 0,6 ms y Naive Bayes 0,6 ms.

## 3. Hallazgo: aprender de una sola cámara no generaliza

En test, los modelos aprendidos rinden parecido a las reglas. En la cámara no vista **todos se derrumban**
(F1 de 0,34 a 0,45), igual que la CNN (0,58), mientras las reglas se mantienen (0,92). El problema no es la CNN en
particular: es **aprender la decisión con datos de una sola cámara**.

**Diagnóstico (no se usó para elegir modelos).** Los descriptores **absolutos** cambian mucho entre cámaras;
mediana por clase:

| Clase | Cámara | `lap_var` | `gray_std` | `hue_std` | `orb_matches` |
|---|---|---|---|---|---|
| normal | 1 | 452,7 | 30,3 | 39,4 | 219 |
| normal | 2 | 59,5 | 53,3 | 50,8 | 159 |
| desenfocada | 1 | 4,8 | 16,5 | 18,0 | 1 |
| desenfocada | 2 | 4,5 | 44,0 | 29,5 | 2 |
| girada | 1 | 585,3 | 39,9 | 70,3 | 8 |
| girada | 2 | 60,6 | 56,3 | 51,6 | 27 |

Por ejemplo, una desenfocada de la cámara 2 tiene el contraste (`gray_std` = 44) de una normal de la cámara 1. Un
modelo aprendido toma esos valores tal cual y se equivoca. SVM entrenada con distintos grupos de descriptores:

| Descriptores de entrada | Test F1 | Cámara no vista F1 |
|---|---|---|
| Todos | 0,895 | 0,422 |
| Solo relativos a la cámara (coincidencias ORB, inliers, ángulo) | 0,727 | 0,590 |
| Solo absolutos (nitidez, contraste, entropía, color) | 0,836 | 0,366 |

Las reglas generalizan porque su **estructura** separa ambos tipos de señal:

1. Deciden normal o saboteada con señales **relativas a la misma cámara**: coincidencias contra sus propias
   referencias y ángulo. La calibración con 8 referencias de cada cámara las vuelve comparables entre cámaras.
2. Usan los absolutos solo para distinguir entre sabotajes, con umbrales de **margen amplio**: la nitidez de una
   desenfocada es 4,8 en la cámara 1 y 4,5 en la 2, muy por debajo del umbral de 18,9.

**Para la defensa y el informe:** el conocimiento del problema escrito en las reglas (qué señal es relativa a la
cámara y cuál no) vale más que aprender de muchos ejemplos de una sola cámara. Para que un modelo aprendido
generalice harían falta datos de varias cámaras o descriptores normalizados por cámara.
