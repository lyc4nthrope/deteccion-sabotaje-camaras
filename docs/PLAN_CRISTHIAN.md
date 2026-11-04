# Plan personal: Cristhian Eduardo Osorio Restrepo

Derivado del [plan general](PLAN.md). Los IDs, dependencias y tipos son los mismos; si algo cambia, se cambia
primero en el plan general.

**Fecha de entrega:** ≈ 15 de noviembre de 2026 (aproximada, por confirmar). Prioridad de cada tarea: **P1**
imprescindible, **P2** recomendada, **P3** recortable primero (sección 5 del plan general).

## Tu rol

Eres dueño de la justificación regional, los datos, las métricas, el módulo clásico y la auditoría de robustez
(artificial y física). Estas tareas forman la
**ruta crítica** del proyecto: la división de datos (T14), las métricas (T25) y el clásico (T40–T43) desbloquean
el trabajo de Santiago (CNN, demo, prueba regional) y la auditoría. Prioriza siempre las tareas ⛔.

Símbolos: ⛔ bloqueante · 🟢 no bloqueante · ⏳ espera un tema de clase. Cada tarea termina con verificación,
commit y Pull Request revisado por Santiago.

## Puedes empezar ya

- T01 Leer y acordar el plan (con Santiago).
- T02 Agregar a Santiago como colaborador (urgente: sin esto Santiago no puede subir cambios).
- T03 Configurar el entorno Python.
- T11 Inventario (T10 ya está hecha: el dataset está descargado).
- T25 Módulo de métricas (solo depende de T03).
- T06 Justificación regional (no depende de nada).

## Tu calendario

Día 1 = martes 3 de noviembre. Detalle y verificación de capacidad en la sección 6 del plan general
(≈ 65 h de tareas P1, ≈ 6 h por día).

- **Día 1 (mar 3 nov):** T01, T03 (con Santiago), T02, T10 ✅, T11, T13.
- **Día 2 (mié 4 nov):** T14 a primera hora (Santiago la espera para T60), T15, T19, T25.
- **Día 3 (jue 5 nov):** T30, T31, T32, T33, T37.
- **Día 4 (vie 6 nov):** T35, T40, T41.
- **Día 5 (sáb 7 nov):** T42, T43.
- **Día 6 (dom 8 nov):** T70, T71.
- **Día 7 (lun 9 nov):** T72, T73, T38 (P2; Santiago la espera para T77).
- **Día 8 (mar 10 nov):** T74, T75 (P2), T06.
- **Día 9 (mié 11 nov):** T92; T90 y T93 con Santiago.
- **Día 10 (jue 12 nov):** T100; T94 y T101 con Santiago.
- **Día 11 (vie 13 nov):** T101, T102.
- **Día 12 (sáb 14 nov):** T103 y margen para atrasos.
- **Solo si hay margen:** T34, T36.

## Lista de tareas

### Fase 0. Organización

- [x] **T00 ⛔** Repositorio, estructura y plan inicial.
- [ ] **T01 ⛔ P1** Leer y acordar este plan con Santiago.
  - **Qué hacer:** leer el plan general y los dos planes personales; anotar dudas o cambios en el PR del plan.
  - **Entregable:** acuerdo escrito en el PR.
  - **Cómo verificar:** ambos aprueban el PR.
- [ ] **T02 ⛔ P1** Agregar a Santiago como colaborador y acordar el flujo de ramas (D10).
  - **Qué hacer:** invitarlo en GitHub (Settings → Collaborators); explicar el flujo: rama por tarea, PR a `main`, revisión del otro.
  - **Entregable:** acceso de Santiago al repositorio.
  - **Cómo verificar:** Santiago abre un PR de prueba y tú lo revisas.
  - **Quién te espera:** Santiago, para subir cualquier tarea suya.
- [ ] **T03 🟢 P1** Configurar el entorno Python (local o Colab) con `requirements.txt`.
  - **Qué hacer:** crear un entorno virtual e instalar `requirements.txt`.
  - **Cómo verificar:** `pytest` se ejecuta y `import cv2` funciona.
- [ ] **T06 🟢 P1** Justificación regional en `docs/contexto_regional.md`.
  - **Qué hacer:** verificar y citar fuentes sobre cámaras fuera de servicio o vandalizadas en Colombia y su relevancia en Armenia, Quindío; anotar fecha de consulta. Incluir la nota aprobada por la profesora: no se encontró un dataset público regional de sabotaje de cámaras; Bremen se usó para entender y desarrollar el trabajo; el método se aplica igual a un dataset regional. Citar el dataset de Medellín (D12, Zenodo, DOI 10.5281/zenodo.7935299, CC BY 4.0) como dato regional usado. No dar por ciertas las cifras hasta verificarlas. Fuentes de partida:
    - https://www.bluradio.com/regiones/santanderes/videovigilancia-en-bucaramanga-de-808-camaras-sirven-526-rg10
    - https://www.eltiempo.com/bogota/camaras-de-seguridad-fuera-de-servicio-por-robo-vandalismo-y-obras-706006
    - https://www.bluradio.com/regiones/santanderes/robaron-2-000-metros-de-cable-de-camaras-de-seguridad-en-bucaramanga-rg10
    - https://armenia.gov.co/alcaldia-de-armenia-rechaza-actos-vandalicos-contra-red-semaforica-y-llama-a-la-ciudadania-a-proteger-la-infraestructura-publica
    - https://www.pulzo.com/nacion/robo-de-30-millones-en-armenia-camaras-mapas-de-calor-y-reaccion-policial-permiten-captura-PP5150773A
  - **Entregable:** `docs/contexto_regional.md`.
  - **Cómo verificar:** cada afirmación tiene su fuente verificada.
  - **Quién te espera:** T101 (sección de contexto regional del informe).

### Fase 1. Datos

- [x] **T10 ⛔ P1** Descargar el dataset de Bremen y registrar las condiciones de uso (hecha: descarga completa).
  - **Método:** WebDAV público, `curl -u "GCsBcswrQf9kBXC:" https://nc.uni-bremen.de/public.php/webdav/<ruta>`; 1,46 GB en 2 ubicaciones × 4 zips (Normal, Blur, Rotated, Noimage) en `data/raw/bremen/zips/`; cada zip trae una carpeta superior (por ejemplo `Location1_Normal`) que se extrae en `data/raw/bremen/`.
  - **Entregable:** `data/README.md` actualizado.
- [x] **T11 ⛔ P1** Inventario por ubicación y clase.
  - **Qué hacer:** escribir `src/data.py` que recorra `data/raw/` y genere `results/inventario.csv` (ruta, clase, cámara, resolución, formato).
  - **Entregable:** `src/data.py`, `results/inventario.csv`.
  - **Cómo verificar:** ubicación 1: 1800/600/600/600 (normal/desenfocada/girada/tapada); ubicación 2: 753/260/167/251; total 5031.
  - **Quién te espera:** Santiago (T12, galería).
- [x] **T13 ⛔ P1** Analizar nombres y fechas de archivos y definir el criterio de división por bloques de tiempo dentro de la ubicación 1.
  - **Qué hacer:** revisar si los nombres o metadatos tienen fecha, hora o cámara; identificar secuencias de frames casi idénticos.
  - **Entregable:** nota en `docs/datos.md`.
  - **Cómo verificar:** el criterio de división queda escrito y justificado.
- [x] **T14 ⛔ P1** División 70/15/15 por bloques de tiempo dentro de la ubicación 1.
  - **Qué hacer:** generar `data/splits/split.csv` según el criterio de T13; escribir `load_split` en `src/data.py`; escribir la prueba de no solapamiento en `tests/test_data.py`.
  - **Entregable:** `data/splits/split.csv`, `src/data.py`, `tests/test_data.py`.
  - **Cómo verificar:** `pytest` pasa y ningún bloque aparece en dos conjuntos.
  - **Quién te espera:** Santiago (T60, carga de datos de la CNN); tú mismo (T30, T70).
- [x] **T15 ⛔ P1** Imágenes de referencia normales por cámara (para ORB).
  - **Qué hacer:** elegir varias normales por cámara, solo de train, con distintas horas y climas si es posible; listarlas.
  - **Entregable:** lista en `docs/datos.md`.
  - **Cómo verificar:** cada cámara tiene referencias y todas pertenecen a train.
- [x] **T19 ⛔ P1** Conjunto de prueba de cámara no vista (D6).
  - **Qué hacer:** reservar toda la ubicación 2 (1431 imágenes) como prueba de cámara nunca vista; no se usa para entrenar ni para calibrar umbrales. Mide la generalización a otra cámara, la limitación que advierte el artículo.
  - **Entregable:** `data/splits/camara_no_vista.csv` y nota en `docs/datos.md`.
  - **Cómo verificar:** el archivo lista las 1431 imágenes y ninguna aparece en `split.csv`.
  - **Quién te espera:** Santiago (T64, evaluación de la CNN); tú mismo (T43, T71).

### Fase 3. Métricas

- [x] **T25 ⛔ P1** Módulo de métricas.
  - **Qué hacer:** en `src/metrics.py`, funciones para matriz de confusión, F1 macro, tasa de falsos positivos (normal predicha como saboteada), ROC binaria con AUC y tiempo por imagen en ms.
  - **Entregable:** `src/metrics.py`, `tests/test_metrics.py`.
  - **Cómo verificar:** pruebas con ejemplos de resultado conocido pasan.
  - **Quién te espera:** Santiago (T64, evaluación de la CNN); tú mismo (T43, T71).

### Fase 4. Módulo clásico: descriptores

- [x] **T30 ⛔ P1** Preprocesamiento en `src/features.py`: escala de grises, tamaño fijo, filtro Gaussiano.
  - **Cómo verificar:** prueba de tamaño fijo y un solo canal.
- [x] **T31 ⛔ P1** Varianza del Laplaciano (nitidez).
  - **Cómo verificar:** una imagen suavizada da menor valor que la original.
- [x] **T32 ⛔ P1** Desviación estándar y entropía del histograma (uniformidad).
  - **Cómo verificar:** una imagen constante da 0 en ambos.
- [x] **T33 ⛔ P1** Descriptor geométrico con la configuración de la clase 11.
  - **Qué hacer:** ORB con `nfeatures=1000`, `BFMatcher(NORM_HAMMING)`, `knnMatch` k=2 y razón de Lowe 0,75 contra las referencias de T15; puntaje = número de buenas coincidencias; ángulo por homografía con RANSAC (viene del artículo, no de la clase).
  - **Cómo verificar:** una imagen rotada un ángulo conocido devuelve un ángulo cercano.
- [ ] **T34 🟢 P2** Densidad de bordes Canny, orientación dominante de Hough y HOG.
  - **Qué hacer:** HOG con scikit-image (`orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`) y distancia L2 frente a las referencias, como señal global de cambio de escena. Cuidado: el notebook de clase calcula la distancia sobre la imagen de visualización (`visualize=True` devuelve `(features, image)`); usar los vectores de características.
  - **Cómo verificar:** prueba sobre una imagen sintética con bordes conocidos.
- [x] **T37 ⛔ P1** Descriptor de color en HSV (umbralización de color).
  - **Qué hacer:** saturación media, dispersión del matiz y proporción de píxeles en rango de tono de piel (mano tapando la cámara); sirve para separar "tapada" de una escena nocturna oscura.
  - **Entregable:** `src/features.py`.
  - **Cómo verificar:** una imagen sintética de tono piel da proporción alta y una gris da saturación cercana a 0.
- [x] **T38 🟢 P2** Comparar ORB, SIFT y KAZE como descriptor geométrico.
  - **Qué hacer:** en train y val, medir acierto en "girada", buenas coincidencias y tiempo por frame con la configuración de clase (KAZE: `NORM_L2`, razón 0,7; SIFT: FLANN KD-tree `trees=5`, `checks=50`, razón 0,7); elegir y justificar (ORB para tiempo real, KAZE para robustez).
  - **Entregable:** sección en `notebooks/03_descriptores.ipynb`, `results/clasico/comparacion_descriptores.csv`.
  - **Quién te espera:** Santiago (T77, robustez de descriptores).
- [x] **T35 ⛔ P1** Análisis de descriptores sobre train (depende de T31, T32, T33 y T37).
  - **Qué hacer:** calcular todos los descriptores sobre train, hacer diagramas de caja por clase y escribir qué descriptor separa cada clase.
  - **Entregable:** `notebooks/03_descriptores.ipynb`.
  - **Cómo verificar:** gráficas y conclusiones escritas.
- [ ] **T36 🟢 P2** Ficha de descriptores (para el informe y la réplica opcional en MATLAB).
  - **Qué hacer:** por cada descriptor, incluido el de color, escribir fórmula, parámetros exactos (tamaño de kernel, umbrales de Canny, número de características ORB, rangos HSV, etc.) y tamaño de imagen.
  - **Entregable:** `docs/descriptores.md`.
  - **Cómo verificar:** cada descriptor se puede reproducir solo con la ficha.
  - **Nota:** solo la usaría Santiago si se hace la Fase 6 opcional (MATLAB); no bloquea a nadie.

### Fase 5. Módulo clásico: reglas y evaluación

- [x] **T40 ⛔ P1** Árbol de reglas en `src/classic.py`: tapada → desenfocada → girada → normal (la regla de tapada puede usar el descriptor de color de T37).
  - **Cómo verificar:** clasifica bien imágenes de ejemplo de cada clase.
  - **Quién te espera:** Santiago (T82, demo en vivo).
- [x] **T41 ⛔ P1** Calibrar umbrales con train.
  - **Qué hacer:** fijar cada umbral con percentiles de las normales de train; justificar con una gráfica; guardar en `results/clasico/umbrales.json`.
  - **Cómo verificar:** cada umbral tiene gráfica y justificación.
- [x] **T42 ⛔ P1** Puntaje continuo de sabotaje.
  - **Qué hacer:** combinar los descriptores normalizados en un puntaje entre 0 y 1 para trazar la ROC.
  - **Cómo verificar:** el puntaje promedio de las saboteadas es mayor que el de las normales.
- [x] **T43 ⛔ P1** Evaluación del clásico.
  - **Qué hacer:** ajustar con val; evaluar una sola vez en test y en la cámara no vista (T19): matriz, F1, tasa de falsos positivos y tiempo por frame.
  - **Entregable:** `results/clasico/`.
  - **Cómo verificar:** métricas de test guardadas.
  - **Quién te espera:** Santiago (T44, análisis de errores; T76, prueba regional; T83, generalización).

### Fase 9. Auditoría de robustez

- [ ] **T70 ⛔ P1** Conjuntos de prueba degradados (test × perturbación × nivel).
  - **Esperas a:** Santiago (T20, perturbaciones).
  - **Cómo verificar:** el número de imágenes coincide con test × 6 × 3 (brillo oscuro y claro por separado).
- [ ] **T71 ⛔ P1** Ejecutar ambos modelos en cada condición: F1, tasa de falsos positivos y tiempo.
  - **Esperas a:** Santiago (T64, CNN evaluada).
  - **Entregable:** `results/auditoria/tabla.csv`.
- [ ] **T72 ⛔ P1** Matrices de confusión cruzadas (modelo × condición crítica).
- [ ] **T73 ⛔ P1** Curvas ROC por condición y F1 frente a severidad.
- [ ] **T74 ⛔ P1** `notebooks/06_auditoria.ipynb` con la interpretación.
  - **Cómo verificar:** cada condición crítica tiene su interpretación (incluida la confusión "oscura ↔ tapada").
  - **Quién te espera:** Santiago (T90, diagnóstico de la CNN).
- [ ] **T75 ⛔ P2** Auditoría física.
  - **Esperas a:** Santiago (T81, conjunto de iluminación física); tu tarea T71.
  - **Qué hacer:** ejecutar ambos modelos sobre el conjunto de iluminación física (luz apagada, lámpara, contraluz) y compararlo con la perturbación artificial equivalente: ¿el oscurecimiento artificial predice la luz baja real?
  - **Entregable:** sección en `notebooks/06_auditoria.ipynb`, `results/auditoria/fisica.csv`.
  - **Quién te espera:** Santiago (T90, diagnóstico de la CNN); tú mismo (T90, clásico).

### Fase 10. Defensa (tu parte)

- [ ] **T90 ⛔ P1** Diagnóstico de los fallos del clásico: qué descriptor se altera con cada perturbación y por qué.
  - **Esperas a:** Santiago (T76, prueba regional en Medellín; T77, robustez de descriptores); tus tareas T74 y T75.
- [ ] **T92 ⛔ P1** Re-evaluar con las correcciones: tabla antes y después.
  - **Esperas a:** Santiago (T91, correcciones).
  - **Entregable:** `results/defensa/`.

## Avisos de bloqueo

| Cuando termines… | Avisa a Santiago porque desbloquea… |
|---|---|
| T02 (colaborador) | Todos sus Pull Requests |
| T11 (inventario) | T12 (galería) |
| T14 (división) | T60 (datos para la CNN) |
| T25 (métricas) | T64 (evaluación de la CNN) |
| T19 (cámara no vista) | T64 (evaluación de la CNN) |
| T38 (comparación de descriptores) | T77 (robustez de descriptores) |
| T40 (reglas) | T82 (demo en vivo) |
| T43 (evaluación del clásico) | T44 (análisis de errores), T76 (prueba regional) y T83 (generalización) |
| T74 y T75 (auditoría artificial y física) | T90 (diagnóstico de la CNN) |
| T92 (antes/después) | T93 (guía de defensa) |

| No empieces… | Hasta que Santiago termine… |
|---|---|
| T70 (conjuntos degradados) | T20 (perturbaciones) |
| T71 (auditoría de ambos modelos) | T64 (CNN evaluada) |
| T92 (antes/después) | T91 (correcciones) |
| T75 (auditoría física) | T81 (captura regional e iluminación física) |
| T93 (guía de defensa) | T83 (generalización) y su parte de T90 |
| T94 (ensayo con la demo) | T85 (filtros en la demo) |
| T90 (diagnóstico del clásico) y T101 (informe) | T76 (prueba regional en Medellín) |
| T90 y T93 (argumento de defensa sobre descriptores) | T77 (robustez de descriptores) |

## Tareas compartidas

- [ ] **T01 ⛔ P1** Acordar el plan (ver Fase 0).
- [ ] **T03 🟢 P1** Entorno Python (ver Fase 0).
- [ ] **T93 ⛔ P1** `docs/defensa.md`: ataque → síntoma → causa matemática → filtro → mejora medida. Tú escribes la parte del clásico.
- [ ] **T94 🟢 P1** Ensayo cruzado con la demo: uno ataca en vivo y el otro explica y activa el filtro, y viceversa (requiere T85 de Santiago).
- [ ] **T100 🟢 P1** Estructura del informe en `informe/` (tuya).
- [ ] **T101 🟢 P1** Tus secciones del informe: contexto regional (a partir de T06), datos, clásico y auditoría.
- [ ] **T102 🟢 P1** Presentación.
- [ ] **T103 ⛔ P1** Revisión cruzada final: revisas las secciones de Santiago.
