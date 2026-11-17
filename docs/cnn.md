# CNN: MobileNetV2 sobre Bremen (T60–T66)

## Arquitectura (T61)
- Base MobileNetV2 preentrenada en ImageNet, entrada 224×224 RGB, escala [-1, 1].
- Cabeza: pooling global, Dropout 0,2 y capa densa de 4 salidas con softmax (D1: normal, desenfocada, girada, tapada).
- Pérdida: entropía cruzada categórica. Optimizador: Adam.
- Pesos por clase balanceados: "girada" tiene solo 84 imágenes de entrenamiento frente a 1120 normales.

## Arquitectura para la defensa (T65)

Todas las cifras salen de `cnn.build_model()` y están fijadas por pruebas (`tests/test_cnn.py`): si el modelo cambia, este texto también debe cambiar.

### Las fórmulas
- **Tamaño de salida de una convolución:** ⌊(n + 2p − f) / s⌋ + 1, con n el tamaño de entrada, p el relleno por lado, f el kernel y s el stride. Con relleno `same` es ⌈n / s⌉.
- **Parámetros:** convolución estándar f² · c_entrada · c_salida (MobileNetV2 no usa sesgo porque normaliza después); **depthwise** f² · c_entrada (un filtro por canal); **pointwise** (1×1) c_entrada · c_salida; **capa densa** entradas · salidas + salidas; **normalización por lote (BN)** 4 · c (γ y β se entrenan; la media y la varianza móviles no).
- **Bloque residual:** y = x + F(x). La suma obliga a que x y F(x) tengan la misma forma.

### Tamaño de salida de las capas con stride 2
Entrada 224×224×3. Hay 5 capas con stride 2: 224 / 2⁵ = 7.

| Capa | Entrada | Salida | Comprobación |
|---|---|---|---|
| `Conv1` (3×3, `same`) | 224×224×3 | 112×112×32 | ⌈224 / 2⌉ = 112 |
| `block_1_depthwise` | 113×113×96 | 56×56×96 | ⌊(113 − 3) / 2⌋ + 1 = 56 |
| `block_3_depthwise` | 57×57×144 | 28×28×144 | ⌊(57 − 3) / 2⌋ + 1 = 28 |
| `block_6_depthwise` | 29×29×192 | 14×14×192 | ⌊(29 − 3) / 2⌋ + 1 = 14 |
| `block_13_depthwise` | 15×15×576 | 7×7×576 | ⌊(15 − 3) / 2⌋ + 1 = 7 |

La entrada de cada depthwise con stride es 1 píxel mayor porque Keras añade un borde de ceros (112 + 1 = 113, y así en los demás). Después, `Conv_1` (1×1, 320 → 1280, 409 600 parámetros) deja 7×7×1280; el pooling global promedia cada mapa a 1 valor (1280), y la cabeza es Dropout 0,2 más una capa densa de 4 salidas con softmax.

### Parámetros (coinciden con `model.summary()`)

| Parte | Parámetros | Estado |
|---|---|---|
| Base MobileNetV2 sin cabeza de ImageNet | 2 257 984 | congelada |
| Cabeza: 1280 × 4 + 4 | **5 124** | **entrenables** |
| Total del modelo | 2 263 108 | |

Solo se entrena el 0,23 % del modelo. Con 1930 imágenes de entrenamiento (84 giradas), ajustar 2,3 millones de parámetros sería sobreajustar; entrenar solo la cabeza equivale a un clasificador lineal sobre 1280 descriptores aprendidos, y por eso converge en pocas épocas con una tasa alta (T66). Ojo con la cifra de la literatura: el MobileNetV2 completo tiene 3 538 984 parámetros, porque su cabeza de 1000 clases de ImageNet suma 1280 × 1000 + 1000 = 1 281 000; aquí no se usa.

Ejemplo de un bloque (`block_2`, entrada y salida de 24 canales, expansión × 6): expansión 1×1 de 24 a 144 canales = 24 · 144 = 3456; depthwise 3×3 sobre 144 canales = 9 · 144 = 1296; proyección 1×1 de 144 a 24 = 3456. Una convolución 3×3 estándar de 144 a 144 tendría 9 · 144 · 144 = 186 624 parámetros: el depthwise más un pointwise 144 → 144 usa 1296 + 20 736 = 22 032, unas 8,5 veces menos.

### Bloques residuales: MobileNetV2 frente a ResNet
MobileNetV2 tiene 17 bloques de residual invertido; 10 llevan la suma `Add` (los de stride 1 con los mismos canales de entrada y salida) y 7 no. Cada bloque hace expansión 1×1 → depthwise 3×3 → proyección 1×1 **lineal** (sin ReLU al final). ResNet usa el orden contrario: reduce con 1×1, convoluciona 3×3 y expande con 1×1, y su atajo une las capas **anchas**; en MobileNetV2 el atajo une las capas **estrechas** (24 canales en el ejemplo), que son las baratas. La idea del atajo es la misma, y ambas redes evitan que el gradiente se desvanezca.

### Por qué no ResNet-50, VGG-16 ni AlexNet
Con la cabeza completa de ImageNet, 224×224, CPU, lote 1. Medido con `python -m src.compare_architectures` (resultado en `results/cnn/arquitecturas.csv`): 3 rondas intercaladas de 30 inferencias cada una; la tabla da la mediana de las rondas y, entre paréntesis, el mínimo y el máximo. Los tiempos dependen de la máquina y de su carga: sirven para comparar los modelos entre sí, no son comparables con el tiempo por cuadro de la auditoría (otro modelo y otro procedimiento). Una corrida aislada puede variar con la carga, por eso se repite en rondas; el orden MobileNetV2 < ResNet-50 < VGG-16 se mantuvo en todas las corridas.

| Modelo | Parámetros | Tiempo por imagen | Frente a MobileNetV2 |
|---|---|---|---|
| MobileNetV2 | 3 538 984 | 82,7 ms (82,4 a 83,3) | 1× |
| ResNet-50 | 25 636 712 | 147,7 ms (147,6 a 148,0) | 1,8× |
| VGG-16 | 138 357 544 | 355,8 ms (351,5 a 370,6) | 4,3× |
| AlexNet | ≈ 61 millones (cifra de la literatura; no está en Keras, no se midió) | | |

- **VGG-16:** 39 veces más parámetros y 4,3 veces más lento; el 74 % de sus parámetros está en una sola capa densa (25 088 × 4096). No tiene conexiones residuales.
- **AlexNet:** es la red más antigua de las tres y tiene capas densas muy grandes; Keras no la trae preentrenada.
- **ResNet-50:** es una alternativa razonable, pero con 7 veces más parámetros y 1,8 veces más lenta en CPU, y la demo en vivo corre en un portátil sin GPU. MobileNetV2 deja tiempo para filtros correctivos y FPS en pantalla.

**Límite:** no se midieron FLOPs ni memoria, solo parámetros y tiempo en CPU en una máquina.

## Datos (T60)
- División de `data/splits/split.csv` (T14). Solo se usan los conjuntos train, val y test. Las filas `descartada` son la banda de guarda (D6) y no entran en ningún conjunto.
- Cámara no vista (ubicación 2) queda fuera del entrenamiento y se usará en T64.

## Entrenamiento (T62)
- Máximo 15 épocas con parada temprana sobre la pérdida de validación (paciencia 3) y restauración de los mejores pesos.
- Curvas de pérdida y accuracy de train y val en `results/cnn/<corrida>/curves.png`.

## Comparación de hiperparámetros (T66)

Cada corrida: `results/cnn/<corrida>/run.json`. Tabla completa en `results/cnn/hiperparametros.csv`.

| Corrida | Tasa | Lote | Capas descongeladas | Mejor época | Pérdida val. | Accuracy val. |
|---|---|---|---|---|---|---|
| lr1e-2_b32_frozen | 1e-2 | 32 | 0 | 3 | 0,0041 | 1,000 |
| lr1e-3_b16_frozen | 1e-3 | 16 | 0 | 15 | 0,0132 | 0,997 |
| lr1e-3_b32_frozen | 1e-3 | 32 | 0 | 15 | 0,0143 | 1,000 |
| lr1e-4_b32_unfreeze10 | 1e-4 | 32 | 10 | 15 | 0,0408 | 0,977 |
| lr1e-4_b32_frozen | 1e-4 | 32 | 0 | 15 | 0,0863 | 0,980 |

**Elección:** tasa 1e-2, lote 32, base congelada. Es la de menor pérdida de validación y converge en 3 épocas. Tasa 1e-4 queda claramente peor: aprende despacio y con 15 épocas no alcanza. Descongelar 10 capas con tasa 1e-4 mejora sobre la base congelada con la misma tasa, pero no supera a 1e-2.

**Limitación:** la validación de la ubicación 1 casi no discrimina. Casi todas las corridas dan 97–100 % de accuracy, porque las imágenes de validación son de la misma cámara y escena que el entrenamiento. La elección se hizo solo con la pérdida de validación de la ubicación 1 (la menor, lr 1e-2). La cámara no vista no se usó para elegir: se evaluó después, en T64.

## Evaluación en test y cámara no vista (T64)

Modelo: lr 1e-2, lote 32, base congelada (la corrida elegida en T66). Métricas de T25, mismas fórmulas que el clásico. Detalle en `results/cnn/evaluacion/`.

| Conjunto | n | Accuracy | F1 macro | FPR normal | AUC binaria |
|---|---|---|---|---|---|
| Test (ubicación 1) | 327 | 0,994 | 0,967 | 0,000 | 0,999 |
| Cámara no vista (ubicación 2) | 1163 | 0,739 | 0,580 | 0,042 | 0,753 |

Mismas métricas para el clásico en la cámara no vista (`results/clasico/metricas.csv`, 1163 imágenes): accuracy 0,947, F1 macro 0,920, FPR normal 0,001, AUC 0,979. Las 8 imágenes de diferencia son las referencias ORB de la cámara no vista (`referencias_camara_no_vista.csv`): el clásico no las evalúa y la CNN sí, hasta esta corrección. Ahora ambos evalúan las mismas 1163.

Matriz de la cámara no vista (filas: real; columnas: predicha):

| Real | normal | desenfocada | girada |
|---|---|---|---|
| normal | 714 | 1 | 30 |
| desenfocada | 110 | 80 | 61 |
| girada | 102 | 0 | 65 |

La CNN acierta en la ubicación 1 pero no generaliza a otra cámara: 110 de 251 desenfocadas y 102 de 167 giradas salen como normal. Es un cambio de dominio, no sobreajuste en el sentido de T63. **Consecuencia:** la elección de hiperparámetros de T66 no basta; la comparación justa con el clásico queda en desventaja para la CNN en este conjunto.

Nota: la cámara no vista no tiene tapadas (las 251 de la ubicación 2 son copias de la ubicación 1, ver docs/datos.md), así que F1 macro usa las tres clases presentes.

## Aumentación y dropout (T63)

Corrida `lr1e-2_b32_frozen_aug_drop05`: misma configuración de T66 con aumentación (volteo horizontal, traslación 5 %, contraste y brillo ±20 %) y dropout 0,5. Resultados en `results/cnn/lr1e-2_b32_frozen_aug_drop05/`.

| Conjunto | Accuracy | F1 macro | FPR normal | AUC |
|---|---|---|---|---|
| Test (ubicación 1) | 0,963 | 0,825 | 0,033 | 0,994 |
| Cámara no vista (ubicación 2) | 0,664 | 0,333 | 0,003 | 0,737 |

**Resultado negativo.** Empeora la cámara no vista: 221 de 251 desenfocadas salen como normal y ninguna girada se acierta. La pérdida de validación también sube (0,26 frente a 0,004 sin aumentación), así que no es un ajuste solo de generalización. No se probaron más variantes: ajustar sobre la cámara no vista la contaminaría como conjunto de prueba. Si se quiere otra variante, la selección debe hacerse solo con validación.

## Sobreajuste (T63)
Con tasa 1e-2 hay señal leve de sobreajuste: la pérdida de validación es mínima en la época 3 y sube después, y la parada temprana corta en la época 6 (pérdida de entrenamiento 0,0002 frente a validación 0,007). Con tasa 1e-3 y lote 32 las dos curvas siguen bajando hasta la época 15 (0,0027 frente a 0,0143): aún no hay sobreajuste, pero la brecha crece. Aumentación y dropout (T63) se aplican solo si la cámara no vista muestra caída.

## Honestidad de la evaluación (nota para la defensa)

- **Elección de hiperparámetros (T66):** se hizo solo con la pérdida de validación de la ubicación 1. La cámara no vista se miró después, en T64.
- **Lo que sí contaminó el examen:** la corrida de aumentación y dropout (T63) se evaluó en la cámara no vista y su resultado (caída a 221 de 251 desenfocadas como normal) decidió no probar más variantes. Por eso el 0,58 de la cámara no vista debe leerse como una cifra ya mirada: es optimista si se compara con una selección hecha a ciegas.
- **Sobreajuste:** con tasa 1e-2 la pérdida de validación sube después de la época 3 mientras la de entrenamiento baja (0,0002 frente a 0,007 al cortar). La parada temprana restaura la época 3, así que el modelo reportado no es el sobreajustado. Con tasa 1e-3 todavía no hay sobreajuste, pero la brecha crece.
- **Atajos:** la validación no detecta que la red aprende el contexto de cada día de grabación (train, val y test comparten días). Es la explicación más probable de la caída en la cámara no vista. Queda pendiente demostrarlo con Grad-CAM (T67).
