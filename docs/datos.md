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
- Cámara: la cámara **no** se conoce con certeza. El número `N` de `imagenN_AAAAMMDD_HHMMSS.jpg` no sirve como grupo (55 grupos, 22 de ellos con 1 a 4 imágenes), así que la cámara se recupera con clustering (sección 7) y el grupo es el **cluster**.
- Criterio: ~600 imágenes en total, repartidas por cluster con un tope por grupo (reparto equitativo), a intervalos regulares en el tiempo dentro de cada grupo, **solo de grupos confiables, nunca una referencia de su grupo y ninguna a menos de 10 minutos de una referencia** (una imagen comparada consigo misma o con una casi idéntica sale «normal» seguro). Una primera versión agrupaba por nombre y dejó 116 de las 600 como referencias (reseña del #31).
- Resultado: 600 imágenes en 72 grupos, sin duplicados: de 5413 candidatas (confiables y no referencia), 492 se descartan por estar a menos de 10 minutos de una referencia, y la selección queda a 10 minutos o más (mediana 166 minutos). Un grupo confiable (el 46) queda sin candidatas. Lista en `data/splits/medellin_normales.csv`; `python -m src.medellin` la regenera desde `medellin_grupos.csv`; una prueba comprueba que no hay referencias en la lista.
- **Pendiente:** excluir imágenes con personas identificables en primer plano. Esto no se ha revisado a mano y no es automatizable de forma fiable; se documenta como limitación antes de publicar cualquier figura.
- Riesgo ORB: el clásico necesita referencias de la misma cámara (T33). Se resuelve con el clustering de la sección 7 (8 referencias por grupo), con los límites que ahí se dicen. La alternativa de la orientación de Hough sin referencias **no está implementada**.

### 6.1 Sabotaje sintético de Medellín (T18)

Código: `src/synthetic_tampering.py`. Cada imagen normal produce 4 variantes (`desenfocada`, `girada`, `tapada_total`, `tapada_parcial`) y un `manifest.csv` con sus parámetros. La semilla de cada imagen sale de su **nombre de archivo** (CRC32), no de su posición en la lista, así que reordenar o filtrar la lista no cambia ninguna variante.

| Variante | Cómo se genera | Por qué |
|---|---|---|
| `desenfocada` | Desenfoque gaussiano con σ de 8 a 14 píxeles del original de 1280×720, más ruido de 0,5 a 0,9 niveles | Con σ de 4 a 8, el clásico llamaba «normal» al 88 % de las desenfocadas (reseña del #2): a 640×360 el desenfoque no quitaba los puntos ORB. Las desenfocadas reales conservan un piso de ruido de sensor |
| `girada` | **Desplazamiento** horizontal de 65 a 90 % del ancho (queda como máximo el 35 % de la vista original) y giro de 0,5° a 6°; la zona nueva se rellena por reflexión | El paneo anterior era un zoom y el clásico lo llamaba «normal» el 92 % de las veces. Los ángulos reales estimados van de 0,5° a 9,7° |
| `tapada_total` | Color realista con degradado suave y ruido de 0,3 a 0,9 niveles | Con ruido de 4 niveles la nitidez daba ~38 (sobre el umbral de 18,9) y el clásico las llamaba «girada». Las tapadas reales tienen una nitidez de 0,9 a 6,5 |
| `tapada_parcial` | Un rectángulo de 85 a 95 % del cuadro con la **misma textura** que la cubierta total | Un rectángulo plano junto a una cubierta con ruido sería una pista trivial para la CNN |

**Cómo se comprobó que se parecen a las reales** (`tests/test_synthetic_realism.py`, se omite si no están las imágenes de Bremen): las variantes de 20 imágenes normales de test de Bremen se miden con las referencias y los umbrales reales del clásico y se comparan con el rango (percentiles 10 a 90) de las imágenes reales de la misma clase: las desenfocadas se llaman «desenfocada» en al menos el 85 % de los casos y con pocas coincidencias ORB; la girada se llama «normal» como máximo el 25 % de las veces (real: 2 % a 16 %), con coincidencias y ángulos en el rango real; las tapadas totales se llaman «tapada» en al menos el 90 % con la nitidez de las reales.

**Límites:**
- **La zona nueva de la girada es un reflejo**, no una escena nueva: tiene textura pero sus puntos no coinciden con la referencia, que es lo que haría una parte de la escena que no se vio. No es una escena real.
- **La `tapada_parcial` casi nunca se llama «tapada» con las reglas del clásico:** el resto de la escena que queda visible es nítido y no coincide con la referencia, así que cae en «girada». Esta clase se evalúa como **sabotaje frente a normal** (la prueba exige que no se llame «normal» más del 10 % de las veces), no por su etiqueta.
- Sigue siendo sabotaje sintético: mide la generalización a otras cámaras, no la apariencia de un sabotaje real.

## 7. Cámaras de Medellín por clustering (T17b)

El nombre de archivo no identifica la cámara (sección 6), así que se recupera con clustering no supervisado
(clase 20, D15). Código: `src/medellin_groups.py`; se reproduce con `python -m src.medellin_groups`.

- **Descriptor:** miniatura en gris de 32×18 estandarizada (media 0, varianza 1). Estandarizar hace que la vista de
  día y la de noche de una misma escena queden cerca; el color y los vehículos (que cambian) pesan poco frente a la
  estructura fija de la calle.
- **Método:** clustering jerárquico de Ward sobre las 6324 imágenes. Dendrograma y curva de altura de fusión en
  `results/figuras/medellin_dendrograma.png`; silhouette y acuerdo con el grupo por nombre para k = 20 a 160 en
  `results/medellin_k.csv`.
- **Número de grupos: k = 80.** Es el número de cámaras que declara el dataset. Los datos no lo imponen: el
  silhouette baja de forma continua con k (0,154 con 20, 0,086 con 80) y la curva de fusiones no tiene una rodilla
  clara cerca de 80 (se aplana desde k ≈ 25). Se eligió 80 porque con 100 o más grupos ya aparecen grupos de menos de
  10 imágenes que hay que descartar (8 imágenes con k = 100, 78 con k = 160), mientras que con 80 no se descarta
  ninguna y la tasa de coincidencia ORB con las referencias propias casi no mejora al subir k: 93,1 % con k = 40,
  94,1 % con 60, 94,5 % con 80, 95,2 % con 100, 96,2 % con 120 y 96,9 % con 160 (columna `tasa_orb_propia` de
  `results/medellin_k.csv`, que `python -m src.medellin_groups` regenera).
- **Referencias:** 8 por grupo, repartidas en el tiempo como en `calibration_split`: 640 en total. Cada grupo
  conserva al menos 10 imágenes; con k = 80 no se descarta ninguna. Lista en `data/splits/medellin_grupos.csv`.
- **Verificación objetiva:** qué parte de las imágenes de cada grupo (hasta 20 por grupo) coincide por ORB con las
  referencias de su propio grupo (`orb_matches` ≥ 12,5, el umbral del clásico). Global: **94,5 %**. Esa cifra sola no
  prueba que los grupos estén separados: hace falta el control negativo de abajo.
- **Verificación visual:** montaje de 12 grupos al azar (semilla 0). La mayoría son una sola cámara, con dos
  problemas: el grupo 52 mezcla escenas distintas y los grupos 2, 4 y 6 parecen ser la misma cámara partida en
  tres. Partir de más es inofensivo (cada trozo conserva sus propias referencias); mezclar cámaras es lo dañino.
  El montaje no se versiona porque muestra peatones (Ley 1581 de 2012): se genera en `data/medellin_grupos_muestra.png`.
- **Grupos dudosos:** 7 grupos tienen una tasa de coincidencia menor de 70 % (11, 25, 47, 49, 52, 61 y 69; 327
  imágenes) y quedan con `confiable = False`. Quedan **73 grupos confiables (5997 imágenes)**. T76 debe usar solo
  `confiable = True`.
- **Control negativo** (`results/medellin_control_negativo.csv` y `medellin_control_pares.csv`, con las 640 parejas
  grupo-otro grupo; `python -m src.medellin_groups --control`): las mismas imágenes (10 por grupo) contra las
  referencias de 5 grupos **cercanos** (centroides más próximos) y de 3 grupos **al azar**, con el mismo umbral de 12,5.

  | Contra | Tasa media de coincidencia | Parejas con 50 % o más |
  |---|---|---|
  | su propio grupo | 0,953 | |
  | 3 grupos al azar | 0,219 | 14,2 % de 240 |
  | 5 grupos cercanos | 0,616 | 61,8 % de 400 |

  Lectura: (1) las referencias propias discriminan (0,95 frente a 0,22 al azar), en ningún grupo la tasa propia es
  menor que su media al azar; (2) pero el umbral de 12,5 **no separa limpio** en escenas de calle: ORB coincide con
  una escena distinta en el 22 % de las imágenes, por calles, rayas y autos parecidos; (3) los clusters cercanos
  suelen ser **la misma cámara partida** (tasa de 1,0 en los pares 2, 6 y 9, en el 19 y el 20 y en el 70 y el 71;
  el 49 % de las parejas cercanas, 197 de 400, coincide al 80 % o más): los 80 clusters **no son 80 cámaras**, son
  más grupos que cámaras. Cota sobre la pureza: si las imágenes de la misma cámara coincidieran siempre
  (tasa 1), la tasa propia de 0,953 y la de 0,219 contra otra cámara implican como máximo un 6 % de imágenes de otra
  cámara por grupo; si coinciden menos, es menos. Es una cota, no una medida.
- **Límite:** no hay una verdad de terreno de la cámara de cada imagen. Los controles acotan la mezcla, no la
  descartan, y solo se miró a mano una muestra de 12 grupos.
