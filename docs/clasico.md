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
coincidencias ORB ≥ match_min  sin ángulo confiable     → girada  (coincidencias sin geometría consistente)
si no, Laplaciano < blur_max:  dispersión del matiz < cover_threshold → tapada; si no → desenfocada
si no                                                  → girada  (imagen nítida de otra escena)
```

El ángulo sale de una transformación de similaridad estimada con RANSAC entre los puntos ORB de la imagen y los
de su mejor referencia; solo se acepta si la apoyan al menos 6 puntos consistentes.

Elementos de la rúbrica que cubre: descriptores matemáticos (Laplaciano, entropía, ORB), transformación
geométrica (similaridad con RANSAC) y umbralización de color (matiz en HSV). El filtrado espacial lineal y no
lineal se aplica en el módulo de correcciones (T91), donde los experimentos de la sección 7 muestran que sí hace
falta.

## 2. Umbrales calibrados con train (T41)

`results/clasico/umbrales.json`. Cada umbral se elige **solo con el conjunto train** de la ubicación 1 (sin las 8
imágenes de referencia, que coincidirían consigo mismas), en el punto medio entre dos valores observados,
maximizando la exactitud balanceada de su decisión (la clase pequeña pesa igual que la grande).

| Umbral | Valor | Cómo se calibra | Interpretación |
|---|---|---|---|
| `match_min` | 12,5 | Normal frente al resto, con `orb_matches` | Con menos de 13 coincidencias contra las referencias, la escena ya no es la normal |
| `blur_max` | 18,9 | Desenfocada y tapada frente a girada, con `lap_var`, **solo entre saboteadas** | Por debajo, la imagen no tiene bordes nítidos |
| `cover_feature` | `hue_std` | El mejor de 5 candidatos (`hue_std`, `gray_std`, `entropy`, `sat_mean`, `lap_var`) | El color, no el brillo, separa una tapa de un desenfoque |
| `cover_threshold` | 5,4° | Tapada frente a desenfocada | Menos de 5,4° de dispersión del matiz: un solo color, **o ningún píxel con color** (imagen gris u oscura; ver sección 6) |
| `angle_max` | 0,94° | Percentil 99 del \|ángulo\| de las normales de train que sí coinciden (≥ `match_min`) | Lo máximo que se mueve una cámara normal (viento, vibración) |

## 3. Puntaje continuo para la curva ROC (T42)

`max(match_min / (match_min + coincidencias), |ángulo| / (|ángulo| + angle_max))`, entre 0 y 1; la parte del
ángulo vale 1 cuando no hay un ángulo confiable. Cada parte vale 0,5 exactamente en su umbral, así que puntaje
> 0,5 equivale a la decisión "saboteada" (verificado en 500 filas aleatorias en las pruebas y en las 2609 filas
reales por una revisión independiente).

## 4. Resultados (T43)

`results/clasico/metricas.csv`, matrices de confusión `matriz_<conjunto>.csv` y puntos ROC `roc_<conjunto>.csv`.

| Conjunto | Imágenes | Exactitud | F1 macro | Falsos positivos (normales) | AUC binaria |
|---|---|---|---|---|---|
| Train | 1930 | 0,941 | 0,898 | 1,8 % | 0,997 |
| Val | 352 | 0,977 | 0,963 | 1,0 % | 0,997 |
| **Test** | 327 | **0,991** | **0,954** | **0,5 %** | **0,997** |
| **Cámara no vista** | 1163 | **0,947** | **0,920** | **0,1 %** | **0,979** |

**Tiempo por frame** (descriptores + reglas sobre la imagen ya decodificada, 100 imágenes de test, con una
llamada de calentamiento descartada): media **13,5 ms**, mediana 10,6 ms (unos 70 cuadros por segundo).

Matriz de confusión de test (filas = real, columnas = predicho):

| | normal | desenfocada | girada | tapada |
|---|---|---|---|---|
| normal | 181 | 0 | 1 | 0 |
| desenfocada | 0 | 68 | 0 | 0 |
| girada | 2 | 0 | 7 | 0 |
| tapada | 0 | 0 | 0 | 68 |

Matriz de confusión de la cámara no vista:

| | normal | desenfocada | girada | tapada |
|---|---|---|---|---|
| normal | 744 | 0 | 1 | 0 |
| desenfocada | 18 | 216 | 17 | 0 |
| girada | 26 | 0 | 141 | 0 |

## 5. Transparencia: test y cámara no vista se evaluaron más de una vez

Los umbrales siempre se calibraron solo con train, pero **las reglas cambiaron dos veces después de ver
resultados**, así que test y la cámara no vista no se evaluaron una sola vez:

| Versión | Qué cambió | Test F1 | Cámara no vista F1 | Giradas detectadas (no vista) |
|---|---|---|---|---|
| 1 | Reglas sin ángulo | 0,913 | 0,630 | 7 de 167 |
| 2 | Regla del ángulo (umbral = percentil 99 de las normales de train) | 0,913 | 0,914 | 135 de 167 |
| 3 | Correcciones de la revisión independiente: ángulo solo con ≥ 6 puntos RANSAC, umbral calculado con normales que coinciden, referencias fuera de la calibración, coincidencias sin geometría = sospechosa | 0,954 | 0,920 | 141 de 167 |

- **Versión 2:** la regla del ángulo se agregó después de ver que, en la cámara no vista, "girada" es un giro
  **parcial** (la escena se superpone: mediana de 27 coincidencias, por encima de `match_min`) y no un cambio
  completo de escena como en la ubicación 1. El umbral se fijó de antemano con las normales de train.
- **Versión 3:** corrige defectos encontrados por una revisión de código (ángulos de 150° calculados con 2 o 3
  puntos inflaban el umbral), no por los resultados.

Por eso los resultados de test y de la cámara no vista son **optimistas**. La prueba de generalización que no se
ha mirado queda para el dataset de Medellín y la captura propia.

## 6. Errores y límites conocidos (para la auditoría y la defensa)

- **Girada con giro pequeño** (2 en test, 26 en la cámara no vista): el ángulo queda por debajo de 0,94°.
- **Desenfocadas nocturnas de la cámara no vista** (18 como normales, 17 como giradas): de noche hay pocos puntos
  ORB y el Laplaciano es bajo también en escenas nítidas.
- **`hue_std` = 0 cuando no hay color:** una imagen gris u oscura sin píxeles de color obtiene 0, igual que una
  tapa de un solo color. Una cámara desenfocada de noche y sin color podría clasificarse como tapada.
- La clase girada de test tiene solo 9 imágenes (ver `docs/datos.md`): su resultado en test es poco estable.

## 7. Experimento: suavizado y ruido en el Laplaciano

Se probó medir la nitidez sobre la imagen con suavizado gaussiano. Exactitud balanceada para separar borrosas
(desenfocadas y tapadas) de nítidas, con 60 imágenes de train por clase y el umbral aprendido sin ruido:

| Variante | Sin ruido | Con ruido gaussiano (σ = 15) |
|---|---|---|
| Laplaciano sin suavizar | 0,996 (umbral 17,5) | 0,500 |
| Gaussiano 3×3 | 1,000 (umbral 2,9) | 0,500 |
| Gaussiano 5×5 | 0,979 (umbral 2,2) | 0,500 |

1. **Ningún suavizado gaussiano protege la regla del ruido:** el ruido crea bordes falsos, el Laplaciano sube y
   una imagen borrosa parece nítida. Es el **fallo que se debe diagnosticar en la defensa**; la corrección
   propuesta es un filtro no lineal (mediana) antes de medir (T91).
2. Sin ruido, el Laplaciano sin suavizar deja más margen (umbral 17,5 frente a 2,9, muy cerca de 0). Por eso el
   módulo lo mide sin suavizar.
