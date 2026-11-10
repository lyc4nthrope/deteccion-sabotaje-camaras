# 3. Datos

## 3.1 Dataset de Bremen

El dataset principal es **ComNets Camera Tampering Images Datasets** de la Universidad de Bremen [6]: imágenes
reales de dos cámaras exteriores, con sabotaje real, etiquetadas en cuatro clases. La ubicación 1 tiene imágenes
de 1280×720 y es la que usa el artículo de referencia; la ubicación 2 tiene imágenes de 4608×2592. El módulo
clásico reduce todas las imágenes a 640×360 antes de procesarlas.

**Tabla 3.1.** Inventario del dataset. Fuente: `results/inventario.csv`.

| Ubicación | Clase | Archivos | Imágenes distintas | Duplicados |
|---|---|---|---|---|
| 1 | normal | 1800 | 1774 | 26 |
| 1 | desenfocada | 600 | 600 | 0 |
| 1 | girada | 600 | **135** | **465** |
| 1 | tapada | 600 | 600 | 0 |
| 2 | normal | 753 | 753 | 0 |
| 2 | desenfocada | 260 | 251 | 9 |
| 2 | girada | 167 | 167 | 0 |
| 2 | tapada | 251 | **0** | **251** |
| | **Total** | **5031** | **4280** | **751** |

Los duplicados se detectaron con el hash MD5 de cada archivo: dos archivos con el mismo hash tienen exactamente
el mismo contenido. Se conserva una copia por contenido.

## 3.2 Hallazgos sobre la calidad del dataset

La revisión del dataset encontró cuatro problemas que, sin corregir, inflarían los resultados:

1. **La clase girada de la ubicación 1 tiene 135 imágenes distintas, no 600.** Las otras 465 son copias con
   nombres como ` - Copy` o `(1)`. El artículo de referencia reporta 600 giradas [6]; si una copia queda en
   entrenamiento y otra igual en prueba, la prueba mide memoria y no generalización. Aquí se eliminan antes de
   dividir.
2. **La clase tapada de la ubicación 2 es una copia de la ubicación 1.** Sus 251 imágenes son idénticas a
   imágenes de la ubicación 1, incluso con los mismos nombres. No son de otra cámara, así que se excluyen de la
   prueba de cámara no vista.
3. **Cada clase se grabó en días distintos.** En la ubicación 1, las normales son del 16 al 18 de julio de 2024,
   las desenfocadas del 18 al 20, las tapadas del 22 y 23 de julio y las giradas del 24 y 25 de agosto. Un modelo
   podría aprender la fecha (luz, clima, vegetación) en lugar del sabotaje. La prueba en otra cámara (sección
   3.4) sirve para detectar ese sesgo, que se discute en la sección 10.
4. **Los ritmos de captura son distintos.** Las normales se tomaron una por minuto y las tapadas en ráfagas de
   una por segundo, así que imágenes consecutivas son casi idénticas. Esto obliga a dividir por bloques de tiempo
   (sección 3.3).

## 3.3 División sin fugas de la ubicación 1

Una división aleatoria pondría dos fotos casi iguales, tomadas con segundos de diferencia, una en entrenamiento y
otra en prueba. Para evitarlo, dentro de cada clase las imágenes se ordenan por fecha y hora y se agrupan en
**bloques de 12 imágenes consecutivas**. Cada bloque completo va a un solo conjunto, sorteado con semilla fija, en
proporción 70/15/15. Un bloque de 12 equivale a unos 12 minutos en las normales y a unos 12 segundos en las
ráfagas de tapadas, así que el criterio se adapta a ambos ritmos.

Además, en cada frontera donde cambia el conjunto se **descartan 2 imágenes a cada lado**. Una revisión
independiente encontró que, sin ese descarte, había 22 fronteras en la clase tapada donde dos fotos tomadas con 1
segundo de diferencia quedaban una en entrenamiento y otra en prueba. Primero se probaron bloques de una hora, pero
casi todas las tapadas caen en la misma hora y la división quedó con 550 imágenes en prueba y 40 en entrenamiento.

**Tabla 3.2.** División de la ubicación 1. Fuente: `data/splits/split.csv`.

| Clase | Entrenamiento | Validación | Prueba | Descartadas | Total |
|---|---|---|---|---|---|
| normal | 1120 | 200 | 182 | 272 | 1774 |
| desenfocada | 356 | 68 | 68 | 108 | 600 |
| girada | 84 | 18 | 9 | 24 | 135 |
| tapada | 370 | 66 | 68 | 96 | 600 |
| **Total** | **1930** | **352** | **327** | **500** | **3109** |

**Limitación:** la clase girada queda con solo 9 imágenes de prueba en la ubicación 1. Por eso su evaluación
principal es la cámara no vista, que tiene 167 giradas (sección 3.4), y se reporta el F1 macro, que pesa igual
cada clase.

## 3.4 Cámara no vista: ubicación 2

La ubicación 2 es una cámara distinta, en otro lugar. **Nunca se usa para entrenar, calibrar umbrales ni elegir
modelos**: mide si lo aprendido en una cámara funciona en otra. Tiene 1171 imágenes distintas: 753 normales, 251
desenfocadas y 167 giradas. No tiene tapadas, porque eran copias de la ubicación 1 (hallazgo 2).

El módulo clásico compara cada imagen contra imágenes de referencia de la **misma** cámara. Como en el artículo
de referencia, se usan 8 referencias normales repartidas a lo largo del día (noche, madrugada, mañana y tarde)
[6]. Para la ubicación 1 salen del conjunto de entrenamiento; para la ubicación 2 se toman 8 de sus normales como
paso de calibración, igual que al instalar el sistema en una cámara nueva, y se excluyen de la evaluación. Por
eso la evaluación de la cámara no vista usa 1163 imágenes. Los umbrales siguen siendo los calibrados en la
ubicación 1. Fuentes: `data/splits/camara_no_vista.csv` y `data/splits/referencias.csv`.

## 3.5 Prueba regional en Medellín

*Redacta Santiago (T16–T18): dataset de cámaras CCTV públicas de Medellín [7], agrupación por cámara (T17b) y
sabotaje sintético.*

## 3.6 Captura propia en Armenia

*Redacta Santiago (T80–T81): protocolo, conteo por clase, conjunto de iluminación física y cumplimiento de la
Ley 1581 de 2012.*
