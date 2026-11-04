# CNN: MobileNetV2 sobre Bremen (T60–T66)

## Arquitectura (T61)
- Base MobileNetV2 preentrenada en ImageNet, entrada 224×224 RGB, escala [-1, 1].
- Cabeza: pooling global, Dropout 0,2 y capa densa de 4 salidas con softmax (D1: normal, desenfocada, girada, tapada).
- Pérdida: entropía cruzada categórica. Optimizador: Adam.
- Pesos por clase balanceados: "girada" tiene solo 84 imágenes de entrenamiento frente a 1120 normales.

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

**Limitación:** la validación de la ubicación 1 casi no discrimina. Casi todas las corridas dan 97–100 % de accuracy, porque las imágenes de validación son de la misma cámara y escena que el entrenamiento. La elección de hiperparámetros se confirmará con la cámara no vista (T64), no con esta tabla.

## Sobreajuste (T63)
Pendiente de decidir con T64. Con tasa 1e-2 hay señal leve de sobreajuste: la pérdida de validación es mínima en la época 3 y sube después, y la parada temprana corta en la época 6 (pérdida de entrenamiento 0,0002 frente a validación 0,007). Con tasa 1e-3 y lote 32 las dos curvas siguen bajando hasta la época 15 (0,0027 frente a 0,0143): aún no hay sobreajuste, pero la brecha crece. Aumentación y dropout (T63) se aplican solo si la cámara no vista muestra caída.
