# Datos: inventario, limpieza y división

Tareas T11, T13, T14, T15 y T19. Todo se genera con `python -m src.data` a partir de `data/raw/bremen/`
(semilla 42). Los archivos de división están versionados en `data/splits/`.

## 1. Inventario (T11)

Archivo: `results/inventario.csv`. Imágenes JPG: la ubicación 1 es de 1280×720; la ubicación 2 es de **4608×2592** (1180 archivos), salvo sus 251 tapadas de 1280×720, que son las copias de la ubicación 1. El módulo clásico reduce todas a 640×360, así que cualquier perturbación debe aplicarse **después** de reducir (de lo contrario el ruido se promedia y casi desaparece).

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

## 2. Hallazgos sobre la calidad del dataset (T13)

1. **Duplicados exactos.** Hay archivos con el mismo contenido byte a byte, nombrados con sufijos como
   ` - Copy`, `(1)`, ` - Copy (2)`. Se detectan con el hash MD5 del archivo y se conserva uno por contenido,
   prefiriendo el nombre sin sufijo de copia.
2. **La clase "girada" de la ubicación 1 tiene 135 imágenes distintas, no 600.** Las otras 465 son copias.
   El artículo de referencia reporta 600 giradas; si se entrenó y evaluó con las copias, imágenes idénticas
   pudieron quedar en entrenamiento y en prueba a la vez, lo que inflaría su resultado. En este proyecto se
   eliminan antes de dividir.
3. **La clase "tapada" de la ubicación 2 es una copia de la ubicación 1.** Sus 251 imágenes son idénticas a
   imágenes de la ubicación 1 (incluso con los mismos nombres). No son de otra cámara, así que se excluyen
   del conjunto de cámara no vista.
4. **Cada clase se grabó en días distintos.** En la ubicación 1: normales del 16 al 18 de julio de 2024,
   desenfocadas del 18 al 20, tapadas del 22 y 23 de julio, y giradas del 24 y 25 de agosto. Un modelo podría
   aprender diferencias de fecha (luz, clima, vegetación) en lugar del sabotaje. La prueba con la cámara no
   vista (ubicación 2), el dataset de Medellín y la captura propia sirven para detectar ese sesgo; se debe
   discutir en el informe.
5. **Ritmos de captura distintos.** Las normales se tomaron cada minuto; las tapadas, en ráfagas de una foto
   por segundo. Imágenes consecutivas son casi idénticas.
6. **Nombres irregulares.** Algunos archivos tienen los segundos truncados a un dígito
   (`image_2024-08-24_19-06-3 (1).jpg`); el lector de fechas acepta uno o dos dígitos.

## 3. División de la ubicación 1 (T14)

Archivo: `data/splits/split.csv` (columnas `path`, `location`, `label`, `timestamp`, `md5`, `block`, `set`).
Se lee con `src.data.load_split()`.

**Criterio:** dentro de cada clase, las imágenes se ordenan por fecha y hora y se agrupan en **bloques de 12
imágenes consecutivas**. Cada bloque completo va a un solo conjunto, sorteado con semilla fija (un generador
por clase), en proporción 70/15/15. Además, en cada frontera donde cambia el conjunto se **descartan 2 imágenes
a cada lado** (`set = "descartada"`): así, dos imágenes vecinas en el tiempo nunca quedan en conjuntos distintos.
Las descartadas no se usan en ningún conjunto. Si una clase tiene menos de 3 bloques, `make_split` falla con un
error en lugar de dejarla sin prueba.

**Por qué bloques de imágenes consecutivas y no división aleatoria:** dos fotos tomadas con un minuto (o un
segundo) de diferencia son casi iguales. Si una cae en entrenamiento y la otra en prueba, la prueba mide
memoria y no generalización. Un bloque de 12 imágenes equivale a unos 12 minutos en las normales y a unos 12
segundos en las ráfagas de tapadas, así que el criterio se adapta a ambos ritmos.

**Por qué no bloques de una hora:** se probó primero, pero las tapadas casi todas caen en la misma hora (ráfaga)
y la división quedó con 550 imágenes en prueba y 40 en entrenamiento. Una prueba sobre el dataset real
(`test_real_split_is_balanced_for_every_class`) verifica ahora que cada clase quede entre 60 % y 80 % en train.

**Por qué el descarte en las fronteras:** una revisión independiente encontró que, sin descarte, había 22
fronteras en la clase tapada donde dos fotos tomadas con 1 segundo de diferencia quedaban una en train y otra
en test.

| Clase | Train | Val | Test | Descartadas | Bloques |
|---|---|---|---|---|---|
| normal | 1120 | 200 | 182 | 272 | 148 |
| desenfocada | 356 | 68 | 68 | 108 | 50 |
| girada | 84 | 18 | 9 | 24 | 12 |
| tapada | 370 | 66 | 68 | 96 | 50 |

**Limitación:** la clase girada tiene solo 135 imágenes distintas y quedan **9 en test**. Su evaluación principal
es la cámara no vista (167 giradas, sección 4). Se reporta F1 macro y se recomienda aumentación con rotaciones
sintéticas al entrenar la red (T63).

## 4. Cámara no vista: ubicación 2 (T19)

Archivo: `data/splits/camara_no_vista.csv`. **1171 imágenes**: 753 normales, 251 desenfocadas y 167 giradas.
Nunca se usan para entrenar ni para calibrar umbrales; miden si lo aprendido funciona en otra cámara. No tiene
clase tapada (sus imágenes eran copias de la ubicación 1); esa clase se evalúa en el test de la ubicación 1, en
Medellín (sintético) y en la captura propia.

## 5. Imágenes de referencia para ORB (T15)

Archivo: `data/splits/referencias.csv`. **8 imágenes normales del conjunto train**, repartidas a lo largo de las
fechas y horas de captura (de noche, madrugada, mañana, tarde), como en el artículo de referencia, que usa 8
referencias con distintas condiciones de luz.

| # | Fecha y hora |
|---|---|
| 1 | 2024-07-16 18:37 |
| 2 | 2024-07-17 01:29 |
| 3 | 2024-07-17 06:33 |
| 4 | 2024-07-17 11:17 |
| 5 | 2024-07-17 15:30 |
| 6 | 2024-07-17 21:44 |
| 7 | 2024-07-18 02:16 |
| 8 | 2024-07-18 07:40 |

Solo sirven para la cámara de la ubicación 1. Para la ubicación 2, Medellín y la demo en vivo, el método clásico
necesita referencias de esa misma cámara: es un paso de **calibración** que se hace al instalar el sistema (igual
que en la demo, T82). Para evaluar la ubicación 2 se toman 8 de sus normales como referencias con
`calibration_split`, que las excluye de la evaluación; los umbrales siguen siendo los calibrados en la
ubicación 1.

## 6. Subconjunto normal de Medellín (T17)

- Fuente: `PedestrianDataSet.zip` (Zenodo, CC BY 4.0). Inventario en `results/inventario_medellin.csv`: 6324 imágenes de 1280×720.
- Cámara: la cámara **no** se conoce con certeza. Se usa el número `N` de `imagenN_AAAAMMDD_HHMMSS.jpg` como grupo, pero hay 55 grupos y 22 de ellos tienen entre 1 y 4 imágenes.
- Criterio: ~600 imágenes en total, repartidas por grupo con un tope por grupo (reparto equitativo; los grupos pequeños aportan todas sus imágenes). Dentro de cada grupo, las imágenes se toman a intervalos regulares en el tiempo.
- Resultado: 600 imágenes en los 55 grupos, sin duplicados. Lista en `data/splits/medellin_normales.csv`. Código en `src/medellin.py`.
- **Pendiente:** excluir imágenes con personas identificables en primer plano. Esto no se ha revisado a mano y no es automatizable de forma fiable; se documenta como limitación antes de publicar cualquier figura.
- Riesgo ORB: las referencias de la misma cámara (T33) no se pueden garantizar con el nombre de archivo. **Sin resolver:** el clásico necesita referencias de la misma cámara para decidir "normal" (T33). Medellín todavía no tiene referencias por cámara, así que esta selección no basta para T76. Hough sin referencias no está implementado.
