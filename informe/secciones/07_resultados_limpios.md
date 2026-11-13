# 7. Resultados en condiciones limpias

Esta sección reporta ambos módulos sobre imágenes sin perturbar: el conjunto de prueba de la ubicación 1 y la
cámara no vista (ubicación 2). La auditoría con perturbaciones está en la sección 8.

## 7.1 Módulo clásico

**Tabla 7.1.** Métricas globales del módulo clásico. Fuente: `results/clasico/metricas.csv`.

| Conjunto | Imágenes | Exactitud | F1 macro | Falsas alarmas (normales marcadas como saboteadas) | AUC binaria |
|---|---|---|---|---|---|
| Entrenamiento | 1930 | 0,941 | 0,898 | 1,8 % | 0,997 |
| Validación | 352 | 0,977 | 0,963 | 1,0 % | 0,997 |
| **Prueba** | 327 | **0,991** | **0,954** | **0,5 %** | **0,997** |
| **Cámara no vista** | 1163 | **0,947** | **0,920** | **0,1 %** | **0,979** |

El F1 macro pesa igual cada clase, así que no lo dominan las normales, que son la mayoría. La tasa de falsas
alarmas es la métrica que pide la rúbrica para entornos estables: en la cámara no vista, solo 1 de 745 imágenes
normales disparó una alarma.

**Tabla 7.2.** Matriz de confusión en prueba (filas = clase real, columnas = predicha). Fuente:
`results/clasico/matriz_test.csv`.

| | normal | desenfocada | girada | tapada |
|---|---|---|---|---|
| normal | 181 | 0 | 1 | 0 |
| desenfocada | 0 | 68 | 0 | 0 |
| girada | 2 | 0 | 7 | 0 |
| tapada | 0 | 0 | 0 | 68 |

**Tabla 7.3.** Matriz de confusión en la cámara no vista. Fuente: `results/clasico/matriz_camara_no_vista.csv`.

| | normal | desenfocada | girada |
|---|---|---|---|
| normal | 744 | 0 | 1 |
| desenfocada | 18 | 216 | 17 |
| girada | 26 | 0 | 141 |

**Tabla 7.4.** Métricas por clase (uno contra el resto). Calculadas con `src/metrics.per_class_report` a partir de
las matrices de las tablas 7.2 y 7.3; su F1 macro coincide con el de la tabla 7.1.

| Conjunto | Clase | Precisión | Recall | Especificidad | F1 | Imágenes |
|---|---|---|---|---|---|---|
| Prueba | normal | 0,989 | 0,995 | 0,986 | 0,992 | 182 |
| Prueba | desenfocada | 1,000 | 1,000 | 1,000 | 1,000 | 68 |
| Prueba | girada | 0,875 | 0,778 | 0,997 | 0,824 | 9 |
| Prueba | tapada | 1,000 | 1,000 | 1,000 | 1,000 | 68 |
| Cámara no vista | normal | 0,944 | 0,999 | 0,895 | 0,971 | 745 |
| Cámara no vista | desenfocada | 1,000 | 0,861 | 1,000 | 0,925 | 251 |
| Cámara no vista | girada | 0,887 | 0,844 | 0,982 | 0,865 | 167 |

**Lectura.** Los errores no se reparten al azar; salen de las ramas del árbol que las decidieron:

1. **Giros pequeños** (2 giradas en prueba y 26 en la cámara no vista, predichas como normales): la cámara giró
   menos que `angle_max` (0,94°), el umbral que tolera la vibración normal. Bajarlo convertiría el viento en
   falsas alarmas.
2. **Desenfocadas de la cámara no vista** (18 como normales y 17 como giradas), por dos ramas distintas del árbol
   (análisis de errores, T44):
   - **18 como normales:** el desenfoque no rompe la coincidencia ORB (mediana de 66,5 coincidencias, muy sobre
     `match_min` = 12,5) y su giro queda bajo `angle_max`, así que el árbol las toma por la misma vista. Es la
     misma causa que en el punto 1: la regla de coincidencias y ángulo se evalúa antes que la nitidez.
   - **17 como giradas:** 12 por nitidez (`lap_var` ≥ `blur_max`, aunque la imagen esté borrosa), 3 por giro grande
     y 2 porque no hay ángulo confiable.

La clase girada de prueba tiene solo 9 imágenes: un error cambia su recall en 11 puntos. Por eso su resultado
principal es el de la cámara no vista, con 167 giradas.

**Tiempo por cuadro.** Descriptores y reglas sobre la imagen ya decodificada, medidos en 100 imágenes de prueba
con una llamada de calentamiento descartada: media de **12,6 ms** y mediana de **10,4 ms**, unos 79 cuadros por
segundo con la media. Basta para una cámara en tiempo real. Fuente: `results/clasico/tiempo.json`.

**Curva ROC.** El puntaje continuo (sección 5.5) da un AUC de 0,997 en prueba y de 0,979 en la cámara no vista.
Los puntos de cada curva están en `results/clasico/roc_<conjunto>.csv`; la figura se genera con la auditoría
(T73), junto a las curvas de cada perturbación, para compararlas en un mismo gráfico.

## 7.2 Módulo de aprendizaje profundo

*Redacta Santiago (T64): métricas, matrices de confusión, curva ROC y tiempo por cuadro de la CNN en prueba y en
la cámara no vista, con las mismas funciones de `src/metrics.py`.*

## 7.3 Comparación con los clasificadores aprendidos

**Tabla 7.5.** F1 macro de las reglas frente a KNN, SVM y Naive Bayes entrenados sobre los mismos descriptores
(sección 5.8). Fuentes: `results/clasico/metricas.csv` y `results/clasico/clasificadores.csv`.

| Modelo | Validación | Prueba | Cámara no vista | Falsas alarmas (no vista) |
|---|---|---|---|---|
| **Reglas** | 0,963 | **0,954** | **0,920** | **0,1 %** |
| KNN (k = 5, Manhattan) | 0,948 | 0,888 | 0,451 | 6,8 % |
| SVM (lineal, C = 1) | 0,974 | 0,895 | 0,422 | 28,3 % |
| Naive Bayes | 0,852 | 0,919 | 0,339 | 3,8 % |

En prueba, los tres modelos aprendidos rinden parecido a las reglas. En la cámara no vista **todos se derrumban**
(F1 entre 0,34 y 0,45) y las reglas se mantienen (0,92). La causa es que los descriptores **absolutos** cambian
entre cámaras: una desenfocada de la cámara 2 tiene el contraste (`gray_std` mediano de 44) de una normal de la
cámara 1 (30). Un modelo aprendido toma esos valores tal cual. Las reglas, en cambio, deciden sobre todo con
**comparaciones contra referencias de la misma cámara** (coincidencias ORB y ángulo), que no dependen de la escena.
Este análisis se hizo después de evaluar y no se usó para elegir modelos. Detalle en `docs/ml_clasico.md`,
sección 3.
