# Módulo clásico: reglas, umbrales y resultados

Tareas T40, T41, T42 y T43. Código: `src/features.py` (descriptores), `src/classic.py` (reglas),
`src/evaluate_classic.py` (evaluación). Reproducir desde la raíz del repositorio:

```bash
python -m src.data            # división de datos
python -m src.build_features  # descriptores de todas las imágenes (≈ 2 min)
python -m src.evaluate_classic
```

## 1. Reglas (T40)

```
coincidencias ORB ≥ match_min  y |ángulo| ≤ angle_max  → normal
coincidencias ORB ≥ match_min  y |ángulo| >  angle_max  → girada  (misma escena, vista girada)
si no, Laplaciano < blur_max:  dispersión del matiz < cover_threshold → tapada; si no → desenfocada
si no                                                  → girada  (imagen nítida de otra escena)
```

Cubre los elementos que pide la rúbrica para el módulo clásico: descriptores matemáticos (Laplaciano, entropía,
ORB), transformación geométrica (similaridad estimada con RANSAC para el ángulo), umbralización de color (matiz en
HSV) y filtrado espacial lineal (suavizado gaussiano en el preprocesamiento).

## 2. Umbrales calibrados con train (T41)

`results/clasico/umbrales.json`. Cada umbral se elige **solo con el conjunto train** de la ubicación 1, en el punto
medio entre dos valores observados, maximizando la exactitud balanceada de su decisión (la clase pequeña pesa igual
que la grande).

| Umbral | Valor | Cómo se calibra | Interpretación |
|---|---|---|---|
| `match_min` | 12,5 | Normal frente al resto, con `orb_matches` | Con menos de 13 coincidencias contra las referencias, la escena ya no es la normal |
| `blur_max` | 18,9 | Desenfocada y tapada frente a girada, con `lap_var`, **solo entre saboteadas** | Por debajo, la imagen no tiene bordes nítidos |
| `cover_feature` | `hue_std` | El mejor de 5 candidatos (`hue_std`, `gray_std`, `entropy`, `sat_mean`, `lap_var`) | El color, no el brillo, separa una tapa de un desenfoque |
| `cover_threshold` | 5,4° | Tapada frente a desenfocada | Menos de 5,4° de dispersión del matiz = un solo color (la tapa) |
| `angle_max` | 1,40° | Percentil 99 del \|ángulo\| de las normales de train | Lo máximo que se mueve una cámara normal (viento, vibración) |

## 3. Puntaje continuo para la curva ROC (T42)

`max(match_min / (match_min + coincidencias), |ángulo| / (|ángulo| + angle_max))`, entre 0 y 1. Cada parte vale
0,5 exactamente en su umbral, así que puntaje > 0,5 equivale a la decisión "saboteada".

## 4. Resultados (T43)

`results/clasico/metricas.csv`, matrices de confusión `matriz_<conjunto>.csv` y puntos ROC `roc_<conjunto>.csv`.

| Conjunto | Imágenes | Exactitud | F1 macro | Falsos positivos (normales) | AUC binaria |
|---|---|---|---|---|---|
| Train | 1930 | 0,944 | 0,904 | 1,3 % | 0,999 |
| Val | 352 | 0,980 | 0,970 | 0,5 % | 1,000 |
| **Test** | 327 | **0,985** | **0,913** | **0,5 %** | **0,999** |
| **Cámara no vista** | 1163 | **0,942** | **0,914** | **0,0 %** | **0,982** |

**Tiempo por frame** (descriptores + reglas sobre la imagen ya decodificada, 100 imágenes de test, con una
llamada de calentamiento descartada): media **13,7 ms**, mediana 10,8 ms (unos 70 cuadros por segundo).

Matriz de confusión de test (filas = real, columnas = predicho):

| | normal | desenfocada | girada | tapada |
|---|---|---|---|---|
| normal | 181 | 0 | 1 | 0 |
| desenfocada | 0 | 68 | 0 | 0 |
| girada | 4 | 0 | 5 | 0 |
| tapada | 0 | 0 | 0 | 68 |

Matriz de confusión de la cámara no vista:

| | normal | desenfocada | girada | tapada |
|---|---|---|---|---|
| normal | 745 | 0 | 0 | 0 |
| desenfocada | 20 | 216 | 15 | 0 |
| girada | 32 | 0 | 135 | 0 |

## 5. Transparencia: la regla del ángulo se agregó después de ver la cámara no vista

La primera versión de las reglas no usaba el ángulo. Con ella, la cámara no vista dio **F1 macro 0,630**: 160 de
167 giradas se clasificaron como normales. La causa: en la ubicación 1 "girada" significa que la cámara apunta a
**otra escena** (casi 0 coincidencias), pero en la ubicación 2 el giro es **parcial** y la escena se superpone
(mediana de 27 coincidencias, por encima de `match_min`). El ángulo estimado sí lo delata (mediana 3,6° frente a
0,1° en las normales).

Se agregó la regla del ángulo, que también usa el artículo de referencia, **calibrando su umbral solo con las
normales de train de la ubicación 1** (percentil 99, fijado de antemano), sin ajustar nada con la cámara no vista.
Aun así, la decisión de agregar la regla se tomó después de ver ese error, así que el resultado de la cámara no vista
(F1 0,914) es optimista frente a una cámara realmente nueva. La prueba definitiva de generalización queda para el
dataset de Medellín y la captura propia, que no se han mirado.

| Versión | Test F1 | Cámara no vista F1 | Giradas detectadas (cámara no vista) |
|---|---|---|---|
| Sin regla del ángulo | 0,913 | 0,630 | 7 de 167 |
| Con regla del ángulo | 0,913 | 0,914 | 135 de 167 |

## 6. Errores que quedan (para la auditoría y la defensa)

- **Girada con poca superposición o giro pequeño** (4 en test, 32 en la cámara no vista): pocas coincidencias de
  diferencia con una normal; es el límite del método.
- **Desenfocadas nocturnas de la cámara no vista** (20 como normales, 15 como giradas): de noche hay pocos puntos
  ORB y el Laplaciano es bajo también en escenas nítidas.
- **Tapadas de val clasificadas como desenfocadas** (6 de 66): tapas que no son de un solo color.
- La clase girada de test tiene solo 9 imágenes (ver `docs/datos.md`): su resultado en test es poco estable.
