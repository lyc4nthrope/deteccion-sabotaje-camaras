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

Se preparó una prueba regional con el dataset público de cámaras CCTV de Medellín [7] (licencia CC BY 4.0). El
dataset no trae sabotaje ni referencias de la misma cámara, y el módulo clásico las necesita (sección 5), así que
se resolvieron las dos carencias por código y se evaluaron los dos módulos (T76, tabla 3.3).

1. **Recuperar la cámara de cada imagen (T17b).** Cada imagen se reduce a una miniatura gris de 32×18, estandarizada,
   y se agrupa con el método jerárquico de Ward. Con k = 80 grupos se conservan los de al menos 10 imágenes, y cada
   uno toma 8 imágenes como referencias. Elegir k no tiene un codo claro: la silueta decrece con k
   (`results/medellin_k.csv`).
2. **Qué tan confiables son los grupos.** Un grupo es confiable si el 70 % o más de sus imágenes coincide por ORB con
   sus propias referencias; quedan 72 grupos. Esa tasa sola no prueba nada, así que se hizo un **control negativo**:
   las mismas imágenes contra otro grupo dan una tasa media de 0,953 con el propio, 0,219 con 3 grupos al azar y
   0,616 con los 5 más cercanos (`results/medellin_control_negativo.csv`). Las referencias propias discriminan, pero
   el umbral de 12,5 coincidencias da una coincidencia falsa con otra escena cerca del 22 % de las veces, y **los 80
   grupos no son 80 cámaras**: varios vecinos son la misma cámara partida.
3. **Selección sin fugas.** Se eligieron 600 imágenes normales de los grupos confiables, sin ninguna de las
   referencias y a más de 10 minutos de cualquiera; una prueba lo verifica sobre los CSV versionados.
4. **Sabotaje sintético (T18).** Se generan tres clases por código con semilla fija: desenfoque gaussiano,
   desplazamiento horizontal con giro (girada) y oclusión total o parcial (tapada). Los parámetros se midieron contra
   el clasificador clásico y contra los rangos de Bremen: la desenfocada se clasifica como tal en 37 de 40 casos, y
   la tapada total en 40 de 40. **Limitación:** la tapada parcial casi siempre sale "girada", porque lo que queda
   visible es nítido; por eso se reporta también si se detectó como sabotaje, sin exigir la clase.
5. **Evaluación (T76).** Se usan las 600 normales y sus 4 variantes (3 000 imágenes). El módulo clásico conserva
   los umbrales calibrados en Bremen y solo cambia las referencias, las 8 de la cámara de cada grupo. La CNN es la
   de Bremen, sin reentrenar. Nada se calibró con Medellín, y la prueba se corrió una sola vez.

**Tabla 3.3.** Prueba cruzada Bremen → Medellín. Fuente: `results/medellin/metricas.csv` y `recall_por_variante.csv`.

| Módulo | F1 macro | Falsas alarmas | AUC binaria | Desenfocada | Girada | Tapada total | Tapada parcial |
|---|---|---|---|---|---|---|---|
| Clásico | 0,782 | 0,132 | 0,972 | 0,937 | 0,947 | 1,000 | 0,000 (detectada 1,000) |
| CNN | 0,404 | 0,143 | 0,801 | 0,087 | 0,207 | 0,997 | 0,273 (detectada 0,498) |

Las cuatro columnas de variantes son la proporción puesta en la clase correcta. Sin la tapada parcial, el F1
es 0,938 para el clásico y 0,485 para la CNN.

**Qué se lee y qué no.**

- **Se repite el resultado de la cámara no vista:** el clásico generaliza a otras cámaras y la CNN no. La CNN deja
  pasar como normal el 79 % de las giradas (476 de 600) y manda a "girada" más de la mitad de las desenfocadas
  (316 de 600). La causa no se diagnosticó: no hay Grad-CAM ni intervención sobre Medellín.
- **Las falsas alarmas no son bajas en ninguno:** 13,2 % (clásico) y 14,3 % (CNN), frente a 0,5 % y 0,0 % en
  Bremen. El clásico falla en pocas cámaras: 7 de los 72 grupos concentran 39 de sus 79 falsas alarmas.
- **La tabla favorece al clásico, y hay que decirlo:** los parámetros del sabotaje sintético se ajustaron midiendo
  el propio módulo clásico (punto 4), así que sus cifras sobre desenfocada y tapada total son optimistas. Medellín
  además mide la generalización a otras cámaras, no la apariencia de un sabotaje real.
- **72 grupos no son 72 cámaras** (punto 2), y una parte de las falsas alarmas puede venir de esa agrupación.

Código: `src/medellin.py`, `src/medellin_groups.py` y `src/synthetic_tampering.py`. Las imágenes con personas no se
versionan ni se publican (Ley 1581 de 2012).

## 3.6 Captura propia en Armenia

Para tener una muestra de sabotaje real en una escena de la región, se definió un protocolo de captura en Armenia,
Quindío (`docs/captura.md`). **Estado: el protocolo y el ayudante de captura están hechos; las fotos no se tomaron
para esta entrega** (sección 11).

- **Qué se planeó capturar:** unas 100 imágenes por clase (normal, desenfocada, girada y tapada, esta última mitad
  total y mitad parcial), repartidas en tres condiciones de luz (apagada, lámpara y contraluz), más 8 referencias
  normales por luz que no entran en la evaluación.
- **Orden de la sesión:** la luz se fija por bloques y dentro de cada bloque se toman las cuatro clases, sin cambiar
  la luz.
- **Evaluación sin fugas:** una ráfaga entera va a un solo conjunto, porque las fotos de una misma ráfaga son casi
  duplicadas.
- **Protección de datos (Ley 1581 de 2012):** no capturar a personas identificables, o pedir autorización escrita
  previa; difuminar con YOLOv8n antes de usar o compartir, revisar a mano la copia difuminada y borrar las
  originales con personas; nunca publicar imágenes con personas, y los datos no se suben al repositorio.
- **Ayudante:** `src/capture.py` toma las ráfagas, nombra los archivos y llena la hoja de sesión. Se probó con
  cámaras simuladas (14 pruebas) y con un video de 1280×720; no se probó con una webcam real.