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

**Tabla 7.6.** Métricas globales de la CNN (MobileNetV2, tasa 1e-2, lote 32, base congelada). Fuente:
`results/cnn/lr1e-2_b32_frozen/evaluacion/resumen.csv`.

| Conjunto | Imágenes | Exactitud | F1 macro | Falsas alarmas | AUC binaria |
|---|---|---|---|---|---|
| **Prueba** | 327 | **0,994** | **0,967** | **0,0 %** | **1,000** (0,99996) |
| **Cámara no vista** | 1163 | **0,739** | **0,580** | **4,2 %** | **0,753** |

**Tabla 7.7.** Matriz de confusión de la CNN en la cámara no vista (filas = clase real). Fuente:
`results/cnn/lr1e-2_b32_frozen/evaluacion/matriz_camara_no_vista.csv`.

| | normal | desenfocada | girada | tapada |
|---|---|---|---|---|
| normal | 714 | 1 | 30 | 0 |
| desenfocada | 110 | 80 | 61 | 0 |
| girada | 102 | 0 | 65 | 0 |
| tapada | 0 | 0 | 0 | 0 |

En la prueba, que es de la misma cámara que el entrenamiento, la CNN no comete falsas alarmas y solo se equivoca en
2 de 9 giradas. En la cámara no vista **110 de 251 desenfocadas y 102 de 167 giradas salen como normal**: es un
cambio de dominio, no el sobreajuste de la sección 6.4. La cámara no vista no tiene tapadas (las de la ubicación 2
son copias de la 1, sección 3), así que el F1 macro usa tres clases.

**Tiempo por cuadro.** Media de **81,8 ms** (mediana 79,5 ms) en CPU, unos 12 cuadros por segundo, frente a 12,6 ms
del clásico (sección 7.1). Las dos cifras salen de procedimientos distintos (12,6 ms incluye la reducción de
tamaño; 81,8 ms parte de imágenes ya reducidas a 224×224), así que no conviene dividirlas. La comparación en las
mismas condiciones y con la misma máquina está en la sección 8.4: unas 4 veces (12,4 frente a 51,9 ms). Fuente:
`results/cnn/lr1e-2_b32_frozen/evaluacion/tiempo.json`.

**Lectura.** En la misma cámara la CNN rinde igual o mejor que el clásico (F1 0,967 frente a 0,954), y con una cámara
nueva el orden se invierte (0,580 frente a 0,920). Esta diferencia es la que justifica la discusión de la sección 10.
No hay curva ROC de la CNN guardada en `results/`: el AUC se calculó con 1 − P(normal) como puntaje de sabotaje.

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
