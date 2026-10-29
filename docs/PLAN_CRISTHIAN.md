# Plan personal: Cristhian Eduardo Osorio Restrepo

Derivado del [plan general](PLAN.md). Los IDs, dependencias y tipos son los mismos; si algo cambia, se cambia
primero en el plan general.

## Tu rol

Eres dueño de la justificación regional, los datos, las métricas, el módulo clásico y la auditoría de robustez
(artificial y física). Estas tareas forman la
**ruta crítica** del proyecto: la división de datos (T14), las métricas (T25) y el clásico (T40–T43) desbloquean
el trabajo de Santiago (MATLAB, CNN, demo) y la auditoría. Prioriza siempre las tareas ⛔.

Símbolos: ⛔ bloqueante · 🟢 no bloqueante · ⏳ espera un tema de clase. Cada tarea termina con verificación,
commit y Pull Request revisado por Santiago.

## Puedes empezar ya

- T01 Leer y acordar el plan (con Santiago).
- T02 Agregar a Santiago como colaborador (urgente: sin esto Santiago no puede subir cambios).
- T03 Configurar el entorno Python.
- T10 Descargar el dataset.
- T25 Módulo de métricas (solo depende de T03).
- T06 Justificación regional (no depende de nada).

## Lista de tareas

### Fase 0. Organización

- [x] **T00 ⛔** Repositorio, estructura y plan inicial.
- [ ] **T01 ⛔** Leer y acordar este plan con Santiago.
  - **Qué hacer:** leer el plan general y los dos planes personales; anotar dudas o cambios en el PR del plan.
  - **Entregable:** acuerdo escrito en el PR.
  - **Cómo verificar:** ambos aprueban el PR.
- [ ] **T02 ⛔** Agregar a Santiago como colaborador y acordar el flujo de ramas (D10).
  - **Qué hacer:** invitarlo en GitHub (Settings → Collaborators); explicar el flujo: rama por tarea, PR a `main`, revisión del otro.
  - **Entregable:** acceso de Santiago al repositorio.
  - **Cómo verificar:** Santiago abre un PR de prueba y tú lo revisas.
  - **Quién te espera:** Santiago, para subir cualquier tarea suya.
- [ ] **T03 🟢** Configurar el entorno Python (local o Colab) con `requirements.txt`.
  - **Qué hacer:** crear un entorno virtual e instalar `requirements.txt`.
  - **Cómo verificar:** `pytest` se ejecuta y `import cv2` funciona.
- [ ] **T06 🟢** Justificación regional en `docs/contexto_regional.md`.
  - **Qué hacer:** verificar y citar fuentes sobre cámaras fuera de servicio o vandalizadas en Colombia y su relevancia en Armenia, Quindío; anotar fecha de consulta. No dar por ciertas las cifras hasta verificarlas. Fuentes de partida:
    - https://www.bluradio.com/regiones/santanderes/videovigilancia-en-bucaramanga-de-808-camaras-sirven-526-rg10
    - https://www.eltiempo.com/bogota/camaras-de-seguridad-fuera-de-servicio-por-robo-vandalismo-y-obras-706006
    - https://www.bluradio.com/regiones/santanderes/robaron-2-000-metros-de-cable-de-camaras-de-seguridad-en-bucaramanga-rg10
    - https://armenia.gov.co/alcaldia-de-armenia-rechaza-actos-vandalicos-contra-red-semaforica-y-llama-a-la-ciudadania-a-proteger-la-infraestructura-publica
    - https://www.pulzo.com/nacion/robo-de-30-millones-en-armenia-camaras-mapas-de-calor-y-reaccion-policial-permiten-captura-PP5150773A
  - **Entregable:** `docs/contexto_regional.md`.
  - **Cómo verificar:** cada afirmación tiene su fuente verificada.
  - **Quién te espera:** T101 (sección de contexto regional del informe).

### Fase 1. Datos

- [ ] **T10 ⛔** Descargar el dataset a `data/raw/` y registrar las condiciones de uso.
  - **Qué hacer:** descargar desde el enlace de `data/README.md`, descomprimir en `data/raw/`, anotar estructura de carpetas y condiciones de uso.
  - **Entregable:** `data/README.md` actualizado.
  - **Cómo verificar:** las imágenes abren y las condiciones de uso quedan escritas.
- [ ] **T11 ⛔** Inventario por clase y cámara.
  - **Qué hacer:** escribir `src/data.py` que recorra `data/raw/` y genere `results/inventario.csv` (ruta, clase, cámara, resolución, formato).
  - **Entregable:** `src/data.py`, `results/inventario.csv`.
  - **Cómo verificar:** los conteos son 1800/600/600/600.
  - **Quién te espera:** Santiago (T12, galería).
- [ ] **T13 ⛔** Analizar nombres y fechas de archivos y definir el criterio de división.
  - **Qué hacer:** revisar si los nombres o metadatos tienen fecha, hora o cámara; identificar secuencias de frames casi idénticos.
  - **Entregable:** nota en `docs/datos.md`.
  - **Cómo verificar:** el criterio de división queda escrito y justificado.
- [ ] **T14 ⛔** División 70/15/15 por bloques.
  - **Qué hacer:** generar `data/splits/split.csv` según el criterio de T13; escribir `load_split` en `src/data.py`; escribir la prueba de no solapamiento en `tests/test_data.py`.
  - **Entregable:** `data/splits/split.csv`, `src/data.py`, `tests/test_data.py`.
  - **Cómo verificar:** `pytest` pasa y ningún bloque aparece en dos conjuntos.
  - **Quién te espera:** Santiago (T60, carga de datos de la CNN); tú mismo (T30, T70).
- [ ] **T15 ⛔** Imágenes de referencia normales por cámara (para ORB).
  - **Qué hacer:** elegir varias normales por cámara, solo de train, con distintas horas y climas si es posible; listarlas.
  - **Entregable:** lista en `docs/datos.md`.
  - **Cómo verificar:** cada cámara tiene referencias y todas pertenecen a train.

### Fase 3. Métricas

- [ ] **T25 ⛔** Módulo de métricas.
  - **Qué hacer:** en `src/metrics.py`, funciones para matriz de confusión, F1 macro, tasa de falsos positivos (normal predicha como saboteada), ROC binaria con AUC y tiempo por imagen en ms.
  - **Entregable:** `src/metrics.py`, `tests/test_metrics.py`.
  - **Cómo verificar:** pruebas con ejemplos de resultado conocido pasan.
  - **Quién te espera:** Santiago (T64, evaluación de la CNN); tú mismo (T43, T71).

### Fase 4. Módulo clásico: descriptores

- [ ] **T30 ⛔** Preprocesamiento en `src/features.py`: escala de grises, tamaño fijo, filtro Gaussiano.
  - **Cómo verificar:** prueba de tamaño fijo y un solo canal.
- [ ] **T31 ⛔** Varianza del Laplaciano (nitidez).
  - **Cómo verificar:** una imagen suavizada da menor valor que la original.
- [ ] **T32 ⛔** Desviación estándar y entropía del histograma (uniformidad).
  - **Cómo verificar:** una imagen constante da 0 en ambos.
- [ ] **T33 ⛔** Coincidencias ORB contra las referencias de T15 y ángulo por homografía.
  - **Cómo verificar:** una imagen rotada un ángulo conocido devuelve un ángulo cercano.
- [ ] **T34 🟢** Densidad de bordes Canny y orientación dominante de Hough.
  - **Cómo verificar:** prueba sobre una imagen sintética con bordes conocidos.
- [ ] **T37 ⛔** Descriptor de color en HSV (umbralización de color).
  - **Qué hacer:** saturación media, dispersión del matiz y proporción de píxeles en rango de tono de piel (mano tapando la cámara); sirve para separar "tapada" de una escena nocturna oscura.
  - **Entregable:** `src/features.py`.
  - **Cómo verificar:** una imagen sintética de tono piel da proporción alta y una gris da saturación cercana a 0.
- [ ] **T35 ⛔** Análisis de descriptores sobre train (depende de T31, T32, T33 y T37).
  - **Qué hacer:** calcular todos los descriptores sobre train, hacer diagramas de caja por clase y escribir qué descriptor separa cada clase.
  - **Entregable:** `notebooks/03_descriptores.ipynb`.
  - **Cómo verificar:** gráficas y conclusiones escritas.
- [ ] **T36 ⛔** Ficha de descriptores para MATLAB.
  - **Qué hacer:** por cada descriptor, incluido el de color, escribir fórmula, parámetros exactos (tamaño de kernel, umbrales de Canny, número de características ORB, rangos HSV, etc.) y tamaño de imagen.
  - **Entregable:** `docs/descriptores.md`.
  - **Cómo verificar:** Santiago puede replicar cada descriptor solo con la ficha.
  - **Quién te espera:** Santiago (T51, T52, T53).

### Fase 5. Módulo clásico: reglas y evaluación

- [ ] **T40 ⛔** Árbol de reglas en `src/classic.py`: tapada → desenfocada → girada → normal (la regla de tapada puede usar el descriptor de color de T37).
  - **Cómo verificar:** clasifica bien imágenes de ejemplo de cada clase.
  - **Quién te espera:** Santiago (T82, demo en vivo).
- [ ] **T41 ⛔** Calibrar umbrales con train.
  - **Qué hacer:** fijar cada umbral con percentiles de las normales de train; justificar con una gráfica; guardar en `results/clasico/umbrales.json`.
  - **Cómo verificar:** cada umbral tiene gráfica y justificación.
  - **Quién te espera:** Santiago (T53, clasificador en MATLAB).
- [ ] **T42 ⛔** Puntaje continuo de sabotaje.
  - **Qué hacer:** combinar los descriptores normalizados en un puntaje entre 0 y 1 para trazar la ROC.
  - **Cómo verificar:** el puntaje promedio de las saboteadas es mayor que el de las normales.
- [ ] **T43 ⛔** Evaluación del clásico.
  - **Qué hacer:** ajustar con val; evaluar una sola vez en test: matriz, F1, tasa de falsos positivos y tiempo por frame.
  - **Entregable:** `results/clasico/`.
  - **Cómo verificar:** métricas de test guardadas.
  - **Quién te espera:** Santiago (T44, análisis de errores; T83, generalización).

### Fase 9. Auditoría de robustez

- [ ] **T70 ⛔** Conjuntos de prueba degradados (test × perturbación × nivel).
  - **Esperas a:** Santiago (T20, perturbaciones).
  - **Cómo verificar:** el número de imágenes coincide con test × 5 × 3.
- [ ] **T71 ⛔** Ejecutar ambos modelos en cada condición: F1, tasa de falsos positivos y tiempo.
  - **Esperas a:** Santiago (T64, CNN evaluada).
  - **Entregable:** `results/auditoria/tabla.csv`.
- [ ] **T72 ⛔** Matrices de confusión cruzadas (modelo × condición crítica).
- [ ] **T73 ⛔** Curvas ROC por condición y F1 frente a severidad.
- [ ] **T74 ⛔** `notebooks/06_auditoria.ipynb` con la interpretación.
  - **Cómo verificar:** cada condición crítica tiene su interpretación (incluida la confusión "oscura ↔ tapada").
  - **Quién te espera:** Santiago (T90, diagnóstico de la CNN).
- [ ] **T75 ⛔** Auditoría física.
  - **Esperas a:** Santiago (T81, conjunto de iluminación física); tu tarea T71.
  - **Qué hacer:** ejecutar ambos modelos sobre el conjunto de iluminación física (luz apagada, lámpara, contraluz) y compararlo con la perturbación artificial equivalente: ¿el oscurecimiento artificial predice la luz baja real?
  - **Entregable:** sección en `notebooks/06_auditoria.ipynb`, `results/auditoria/fisica.csv`.
  - **Quién te espera:** Santiago (T90, diagnóstico de la CNN); tú mismo (T90, clásico).

### Fase 10. Defensa (tu parte)

- [ ] **T90 ⛔** Diagnóstico de los fallos del clásico: qué descriptor se altera con cada perturbación y por qué.
- [ ] **T92 ⛔** Re-evaluar con las correcciones: tabla antes y después.
  - **Esperas a:** Santiago (T91, correcciones).
  - **Entregable:** `results/defensa/`.

## Avisos de bloqueo

| Cuando termines… | Avisa a Santiago porque desbloquea… |
|---|---|
| T02 (colaborador) | Todos sus Pull Requests |
| T11 (inventario) | T12 (galería) |
| T14 (división) | T60 (datos para la CNN) |
| T25 (métricas) | T64 (evaluación de la CNN) |
| T36 (ficha de descriptores, incluido el color) | T51, T52 (MATLAB) |
| T40 (reglas) | T82 (demo en vivo) |
| T41 (umbrales) | T53 (clasificador en MATLAB) |
| T43 (evaluación del clásico) | T44 (análisis de errores) y T83 (generalización) |
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

## Tareas compartidas

- [ ] **T01 ⛔** Acordar el plan (ver Fase 0).
- [ ] **T03 🟢** Entorno Python (ver Fase 0).
- [ ] **T93 ⛔** `docs/defensa.md`: ataque → síntoma → causa matemática → filtro → mejora medida. Tú escribes la parte del clásico.
- [ ] **T94 🟢** Ensayo cruzado con la demo: uno ataca en vivo y el otro explica y activa el filtro, y viceversa (requiere T85 de Santiago).
- [ ] **T100 🟢** Estructura del informe en `informe/` (tuya).
- [ ] **T101 🟢** Tus secciones del informe: contexto regional (a partir de T06), datos, clásico y auditoría.
- [ ] **T102 🟢** Presentación.
- [ ] **T103 ⛔** Revisión cruzada final: revisas las secciones de Santiago.
