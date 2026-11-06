# Plan general del proyecto: detección de sabotaje en cámaras de vigilancia

**Grupo 7 · Seguridad** · Cristhian Eduardo Osorio Restrepo y Santiago Orozco Zuluaga
**Docente:** Ing. Ana Maria Tamayo Ocampo, Mg · **Fecha de entrega:** ≈ 15 de noviembre de 2026 (aproximada, por confirmar)

Planes personales: [Cristhian](PLAN_CRISTHIAN.md) · [Santiago](PLAN_SANTIAGO.md) · Registro de decisiones: [DECISIONES.md](DECISIONES.md)

> Este plan es un documento vivo. Las tareas marcadas con ⏳ dependen de temas que aún no se han visto en clase;
> desde D14 se pueden empezar ya y se ajustan cuando la profesora los explique (ver sección 8).

---

## 1. Cómo leer este plan

| Símbolo | Significado |
|---|---|
| ⛔ | **Bloqueante:** otra tarea espera por esta. Hacerla primero. |
| 🟢 | **No bloqueante:** se puede hacer en cualquier momento después de sus dependencias. |
| ⏳ | **Tema de clase pendiente, ajustable:** se puede empezar ya con estudio propio y se ajusta cuando se vea el tema en clase (D14). Puede ser también bloqueante: ⏳⛔. |
| ✅ | Tarea hecha. |
| 🔄 | Tarea en curso. |
| ⬜ | Tarea pendiente. |
| Opcional | Solo si sobra tiempo al final. No forma parte de la ruta crítica ni de los totales de horas. |
| P1 / P2 / P3 | Prioridad: imprescindible, recomendada o recortable primero (sección 5). |

Columnas de las tablas de tareas:

- **ID:** identificador fijo de la tarea (se usa en ramas, commits y Pull Requests).
- **Tarea:** qué hacer, en concreto.
- **Responsable:** quién la hace.
- **Depende de:** tareas que deben estar terminadas antes de empezar.
- **Tipo:** ⛔, 🟢 o ⏳.
- **Entregable:** archivo o resultado que queda en el repositorio.
- **Verificación:** cómo se comprueba que la tarea quedó bien.
- **Estado:** ✅ o ⬜.

**Regla:** una tarea = un objetivo pequeño (1 a 3 horas), con un entregable concreto y una verificación.
Cada tarea termina así: verificación → commit → Pull Request a `main` revisado por el otro integrante (D10).

---

## 2. Requisitos frente al estado actual

| # | Requisito (guía del proyecto) | Estado actual | Tareas |
|---|---|---|---|
| R1 | Problema regional de clasificación o detección, con temática distinta por grupo | ✅ Definido: clasificar imágenes de cámaras de vigilancia saboteadas (Seguridad). ⬜ Componente regional: no existe un dataset público regional de sabotaje; se entrena con Bremen y se prueba en cámaras CCTV reales de Medellín con sabotaje sintético (D12), más una captura propia en Armenia (D3) y la justificación regional | T06, T16–T18, T76, T80, T81, T83 |
| R2 | Lenguaje: Python, MATLAB o ambos | ✅ Resuelto: Python (MATLAB opcional), confirmado por la profesora (T04) | T04; T50–T54 opcionales |
| R3 | Módulo 1, clásico (20 %): descriptores matemáticos, transformaciones geométricas, umbralización (incluida la umbralización de color), filtrado lineal y no lineal. Métricas: tiempo de cómputo por frame y tasa de falsos positivos en entornos estables | ⬜ Pendiente. Los temas necesarios ya se vieron en clase. La umbralización de color se cubre con un descriptor HSV (T37) | T30–T44 |
| R4 | Módulo 2, aprendizaje profundo (20 %): CNN propia o transfer learning, con hiperparámetros justificados (no elegidos al azar). Métricas: curvas de loss y accuracy en train y val, sin sobreajuste severo | ⏳ Se empieza ya con estudio propio de Keras y se ajusta con el tema de CNN en clase (D14). Los hiperparámetros se eligen con una comparación pequeña y documentada (T66) | T60–T66 |
| R5 | Módulo 3, auditoría de robustez (20 %): ambos sistemas ante entornos degradados de manera artificial o física (entornos dinámicos reales). Métricas: matrices de confusión cruzadas, curvas ROC y F1-score en condiciones críticas | ⬜ Pendiente. Degradación artificial (perturbaciones, T70–T74), física (captura con cambios reales de iluminación, T75) y cambio de dominio Alemania → Colombia (T76) | T20–T25, T70–T76, T81 |
| R6 | Defensa ante ataque adversario en vivo (40 %): diagnosticar matemáticamente el fallo y proponer el filtro correctivo exacto | ⬜ Pendiente. Demo local en el portátil (D11) con calibración de referencias y teclas para activar los filtros correctivos en vivo | T07, T82, T85, T90–T94 |
| R7 | Dataset | ✅ Bremen descargado: 5031 imágenes de 2 cámaras con sabotaje real (ubicación 1 con 3600, las del artículo; ubicación 2 con 1431), aceptado por la profesora. La ubicación 2 se reserva como prueba de cámara no vista (D6). ✅ Identificado el dataset regional de Medellín (CC BY 4.0) para la prueba cruzada. ⬜ Captura propia en Armenia, Quindío | T10–T19, T80, T81 |

**Brecha principal:** el artículo de referencia compara reglas frente a una CNN, pero no prueba los modelos ante
cambios de iluminación ni ruido, ni en cámaras de otra región. Ese es el aporte de este proyecto: la auditoría de
robustez (artificial y física), la prueba regional cruzada en cámaras de Medellín, la captura propia en Armenia y
la defensa.

---

## 3. Decisiones

| ID | Decisión | Elección | Por qué |
|---|---|---|---|
| D1 | Clases | 4 clases: normal, desenfocada, girada, tapada. Además se reporta la versión binaria (normal frente a saboteada) | Las 4 clases permiten diagnosticar el tipo de sabotaje; la versión binaria permite una curva ROC directa |
| D2 | Dataset principal | ComNets Camera Tampering Images Datasets (Universidad de Bremen), citando el artículo, **para entrenamiento**. ✅ Aceptado por la profesora (T04) con la condición de que el informe diga que no se encontró un dataset público regional de sabotaje de cámaras, que Bremen se usó para entender y desarrollar el trabajo y que el método funcionaría igual con un dataset regional | Imágenes reales con sabotaje real, etiquetadas y de tamaño manejable. No existe un dataset público regional de sabotaje: la búsqueda en datos.gov.co y de cámaras web públicas de Armenia o Salento no dio resultados |
| D3 | Captura propia en Armenia | Captura con webcam o celular fijo en una escena real de Armenia, Quindío: ~100 imágenes por clase (muestra pequeña de sabotaje regional real) y un conjunto con cambios físicos de iluminación (D7). Se respeta la Ley 1581 de 2012 de protección de datos personales | Sirve para la demo en vivo y su calibración, para la auditoría física (T75) y como muestra de sabotaje real que complementa el sabotaje sintético de Medellín (D12). El artículo advierte que la CNN falla si la cámara cambia |
| D4 | Módulo clásico | Árbol de reglas con umbrales calibrados y justificados. Descriptores: varianza del Laplaciano (nitidez), desviación estándar y entropía del histograma (tapada), geométrico para "girada": ORB con la configuración de la clase 11 (`nfeatures=1000`, `BFMatcher` con distancia Hamming, `knnMatch` con k=2 y razón de Lowe 0,75); el puntaje es el número de buenas coincidencias contra las referencias (como en clase) y el ángulo de rotación sale de la homografía con RANSAC (tomado del artículo, extensión más allá de la clase); descriptor de color en HSV: saturación media, dispersión del matiz y proporción de píxeles en rango de tono de piel (mano tapando la cámara), que ayuda a separar "tapada" de una escena nocturna oscura; complementarios: densidad de bordes Canny, orientación dominante de Hough y HOG (distancia L2 entre vectores de características frente a las referencias, como señal global de cambio de escena). SURF se descarta: está patentado y no está disponible en OpenCV estándar. Preprocesamiento: escala de grises, tamaño fijo, filtro Gaussiano (lineal) y mediana (no lineal) | Sigue el método validado del artículo, usa los temas vistos en clase y cubre lo que pide la rúbrica: descriptores matemáticos, transformaciones geométricas, umbralización (incluida la **umbralización de color**, que el resto de descriptores en gris no cubre) y filtrado lineal y no lineal |
| D5 | Módulo profundo | ✅ Framework decidido: **TensorFlow/Keras** (2026-10-29). Transfer learning con MobileNetV2 y una cabeza de 4 clases; la arquitectura se ajusta a lo que se vea en clase (⏳) | Keras es simple de escribir y de explicar, entrega las curvas de loss y accuracy, corre con GPU gratis en Colab y es el framework del artículo de referencia. El transfer learning converge rápido con pocos datos y reduce el sobreajuste |
| D6 | División de datos | train/val/test 70/15/15 de la **ubicación 1** por **bloques de 12 imágenes consecutivas** por clase, descartando 2 imágenes a cada lado de cada frontera entre conjuntos; antes se eliminan **duplicados** exactos (MD5): 4280 imágenes distintas de 5031 archivos. La **ubicación 2** (1171 imágenes distintas, sin sus tapadas, que eran copias de la ubicación 1) es la prueba de **cámara no vista** (T19). Detalle y cifras en `docs/datos.md` | Los frames consecutivos son casi idénticos: una división aleatoria filtra información entre conjuntos e infla las métricas. La cámara no vista mide la generalización a otra cámara, la limitación que advierte el artículo, y respalda el argumento regional: el método se transfiere entre cámaras |
| D7 | Perturbaciones de la auditoría | **6 tipos**: brillo oscuro y brillo claro **por separado**, corrección gamma, ruido gaussiano, sal y pimienta y compresión JPEG; 3 niveles cada una; semilla fija. **El desenfoque no se usa como perturbación.** Además, degradación **física**: captura propia con cambios reales de iluminación (luz apagada, lámpara, contraluz de ventana) para escenas normales y saboteadas | Luz y ruido son los ataques probables en la defensa. El desenfoque es una clase ("desenfocada"): una imagen normal desenfocada cambiaría de etiqueta legítimamente. Nota: una imagen muy oscura puede parecerse a "tapada"; esa confusión es un hallazgo esperado que se debe diagnosticar. La guía admite degradación "artificial o física" y el título habla de "entornos dinámicos reales": la parte física comprueba si las perturbaciones artificiales predicen lo que pasa con luz real |
| D8 | Lenguaje | **Solo Python; MATLAB opcional al final** (Fase 6, solo si sobra tiempo). Confirmado por la profesora (T04): el proyecto puede hacerse en Python, en MATLAB o en ambos | Ahorra ≈ 12 h (Fase 6). Todo el flujo (CNN, demo en vivo, métricas) ya está en Python, y MATLAB solo duplicaría trabajo. Las horas liberadas se dedican al dataset regional (D12) |
| D9 | Métricas | Las de la rúbrica: tiempo por frame, tasa de falsos positivos, curvas de loss y accuracy, matrices de confusión cruzadas, ROC y F1. El módulo clásico debe producir un **puntaje continuo de sabotaje** para poder trazar su ROC | Son exactamente las exigidas; un clasificador por reglas solo da etiquetas, y la ROC necesita un puntaje |
| D10 | Flujo de trabajo con Git | Una rama por tarea (por ejemplo `feat/T14-division-datos`), Pull Request a `main` y revisión del otro integrante; commits convencionales; `main` siempre ejecutable; nunca se suben datos ni modelos | Permite trabajar en paralelo sin romper el trabajo del otro y deja trazabilidad de cada tarea |
| D11 | Entorno de la demo en vivo | La demo corre **localmente en el portátil** con OpenCV (`VideoCapture`) y TensorFlow en CPU; Colab se usa solo para entrenar | Colab no puede acceder a la webcam en tiempo real. La defensa (40 %) depende de que la demo funcione sin conexión ni retrasos |
| D12 | Prueba regional cruzada | Dataset de Medellín ("Dataset for Detecting Motorcyclists in Pedestrian Areas", Zenodo, CC BY 4.0): se toman imágenes normales y se sintetiza el sabotaje por código (desenfoque gaussiano, rotación, oclusión total y parcial), como en el dataset UHCTD. Se entrena en Bremen y se prueba en Medellín | Son cámaras CCTV reales de Colombia con licencia abierta. Muestra la generalización Alemania → Colombia, que es lo que pidió la profesora. El dataset no tiene sabotaje propio, por eso se sintetiza |
| D13 | Aportes de la clase 11 (SIFT, HOG, KAZE, SURF, ORB), 2026-11-03 | ORB con la configuración de clase como descriptor geométrico por defecto; comparación ORB frente a SIFT frente a KAZE (T38) y su robustez ante perturbaciones (T77); HOG como descriptor global complementario (T34); SURF descartado | Según la clase, ORB es el más rápido y libre de patentes, pero pierde precisión con cambios de iluminación, contraste y ruido; KAZE usa difusión no lineal y es más robusto; SIFT (128 dimensiones) ya no tiene patente. Comparar da evidencia para elegir y un argumento para la defensa |
| D14 | No esperar la clase de CNN, 2026-11-03 | Santiago estudia por su cuenta transfer learning con Keras; T60–T66 pasan a "⏳ ajustable": se empiezan ya y se ajustan cuando se vea el tema en clase. La arquitectura puede cambiar | El equipo empezó tarde; esperar el tema dejaría la CNN y todo lo que depende de ella (T71, T76, T84) para el final |

---

## 4. Ruta crítica

La ruta crítica es la cadena de tareas bloqueantes que define la duración mínima del proyecto. Un retraso en
cualquiera de ellas retrasa la entrega; por eso se priorizan sobre las tareas 🟢.

```
Cadena principal:
T10 → T11 → T13 → T14 → T30 … T37 → T40 … T43 → T70 … T76 → T90 … T94

Entradas a la cadena:
T25 (métricas) ──────────────────► T43, T64, T71
T20 (perturbaciones) ────────────► T70
T60 … T66 (CNN, ⏳ ajustable) ───► T64 → T71
T19 (cámara no vista) ───────────► T43, T64, T71
T16 → T17 → T18 (Medellín) ─────► T76
T80 → T81 (captura Armenia) ─────► T75, T83
T82 → T85 (demo con filtros) ────► T94
```

- **T25** (métricas) alimenta la evaluación del clásico (T43), de la CNN (T64) y la auditoría (T71).
- **T20** (perturbaciones) alimenta los conjuntos degradados (T70).
- **T16 → T17 → T18** (dataset de Medellín con sabotaje sintético) alimenta la prueba regional cruzada (T76).
- **T80 → T81** (captura en Armenia) alimenta la auditoría física (T75) y la prueba de generalización (T83).
- **T85** (filtros en la demo) es necesaria para el ensayo de la defensa (T94).
- **T19** (prueba de cámara no vista, ubicación 2) entra en las evaluaciones del clásico, de la CNN y de la auditoría.
- **T60 … T66** se empiezan ya con estudio propio (D14); la arquitectura puede cambiar cuando se vea el tema de CNN.
- **T38** y **T77** (comparación de descriptores) son P2 y quedan fuera de la ruta crítica.
- La Fase 6 (MATLAB, T05 y T50–T54) es opcional y queda fuera de la ruta crítica.

---

## 5. Ruta mínima y prioridades

Cada tarea tiene una prioridad. Las P1 cubren cada criterio de la rúbrica y nunca se recortan; las P2 suman nota;
las P3 son lo primero que se quita si falta tiempo.

- **P1 Imprescindible** (cubre cada criterio de la rúbrica): T01, T02, T03, T06, T07, T10, T11, T13, T14, T15, T19, T20, T21, T25, T30, T31, T32, T33, T35, T37, T40, T41, T42, T43, T60, T61, T62, T63, T64, T65, T66, T70, T71, T72, T73, T74, T80, T81, T82, T84, T85, T90, T91, T92, T93, T94, T100, T101, T102, T103.
  Datos y división (T10–T15, T19), métricas (T25), clásico (T30–T43), CNN (T60–T66), auditoría artificial
  (T70–T74), captura y demo de la defensa (T80–T85, sin T83), defensa (T90–T94) e informe (T06, T100–T103).
- **P2 Recomendado** (suma nota; recortar solo si falta tiempo):
  - T16, T17, T18, T76: prueba regional en Medellín. La profesora aceptó Bremen con nota, así que es una mejora y no un requisito.
  - T75: auditoría física.
  - T83: generalización a la captura propia.
  - T38, T77: comparación de descriptores ORB, SIFT y KAZE.
  - T34: descriptores complementarios.
  - T36: ficha de descriptores.
  - T44: análisis de errores del clásico.
  - T22, T23: grilla y justificación de niveles de perturbación.
- **P3 Recortable primero:** T12 (galería) y la Fase 6 opcional de MATLAB (T05, T50–T54).

Si se recorta una tarea P2 que figura como dependencia (por ejemplo T75 o T76 para T90), se quita también de
las dependencias de la tarea que la esperaba.

**Recorte progresivo si falta tiempo** (en este orden; nunca se recorta una P1):

1. Quitar las tareas P3.
2. Quitar T77 y luego T38.
3. Reducir Medellín a una muestra de ~200 imágenes.
4. Hacer la auditoría física (T75) con una sola condición de luz.
5. Reducir la captura propia a ~50 imágenes por clase.

---

## 6. Calendario por días

Día 1 = martes 3 de noviembre de 2026; entrega ≈ 15 de noviembre (aproximada, por confirmar). El día 12 es margen
para atrasos y revisión final. Cada día respeta las dependencias de la sección 7.

| Día | Fecha | Cristhian | Santiago | Ambos |
|---|---|---|---|---|
| 1 | mar 3 nov | T02, T10 ✅ (ya descargado), T11, T13 | T07, T20, T21; empezar el estudio de Keras y transfer learning | T01, T03 |
| 2 | mié 4 nov | T14 (a primera hora), T15, T19, T25 | T80, T81 (captura en Armenia, con el conjunto de luz real), T60 (después de T14) | — |
| 3 | jue 5 nov | T30, T31, T32, T33, T37 | T61, T62 | — |
| 4 | vie 6 nov | T35, T40, T41 | T66, T63 | — |
| 5 | sáb 7 nov | T42, T43 | T64, T65, T82 | — |
| 6 | dom 8 nov | T70, T71 | T91, T85, T84 | — |
| 7 | lun 9 nov | T72, T73, T38 (P2) | T16, T17, T18 (P2) | — |
| 8 | mar 10 nov | T74, T75 (P2), T06 | T76 (P2), T77 (P2), T83 (P2) | — |
| 9 | mié 11 nov | T92 | Ajustes de T91 | T90, T93 (al final del día) |
| 10 | jue 12 nov | T100 | — | T94 (ensayo), T101 |
| 11 | vie 13 nov | — | — | T101, T102 |
| 12 | sáb 14 nov | — | — | T103; margen para atrasos |
| — | ≈ dom 15 nov | — | — | **Entrega** |

**Solo si hay margen:** T12 (P3), T22, T23, T34, T36, T44 (P2) y la Fase 6 opcional de MATLAB (T05, T50–T54).

**Conflictos de dependencias resueltos:**

- T60 (día 2) depende de T14: Cristhian hace T14 a primera hora; si no está lista, T60 pasa al día 3 junto con T61.
- T77 depende de T38: T38 se adelantó al día 7 y T74 pasó al día 8, que sigue antes de T90 (día 9).
- T85 depende de T91: el día 6 Santiago hace T91 antes que T85.
- T93 depende de T90 y T92, que se hacen el mismo día 9: T93 va al final del día.

### Verificación de capacidad

Las horas son estimaciones (≈ 2 h por tarea; las de la CNN ≈ 2,5 h). Hay 11 días de trabajo más 1 de margen.

| Integrante | Horas P1 propias | Horas P1 compartidas | Total P1 | Horas por día (11 días) | Con P2 |
|---|---|---|---|---|---|
| Cristhian | ≈ 50 h (25 tareas) | ≈ 15 h | ≈ 65 h | ≈ 6 h/día | ≈ 74 h → ≈ 6,7 h/día |
| Santiago | ≈ 40 h (16 tareas) + ≈ 4 h de estudio de Keras | ≈ 15 h | ≈ 59 h | ≈ 5,4 h/día | ≈ 79 h → ≈ 7,2 h/día |

**Si un día se atrasa, se recorta en este orden:** (1) tareas P3; (2) T77; (3) T38; (4) Medellín reducido a ~200
imágenes; (5) T75 con una sola condición de luz. Nunca se recorta una tarea P1.

---

## 7. Tareas por fase

### Fase 0. Organización

Dejar listo el repositorio, el entorno y los acuerdos de trabajo. Sin esto, ninguno de los dos puede subir
cambios de forma ordenada.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T00 | Crear repositorio, estructura de carpetas y plan inicial | Cristhian | — | ⛔ | Repositorio en GitHub | El repositorio existe en GitHub | ✅ |
| T01 | Leer y acordar este plan entre los dos | Ambos | T00 | ⛔ | Comentario o acuerdo en el PR del plan | Ambos aprueban | ⬜ |
| T02 | Agregar a Santiago como colaborador y acordar el flujo de ramas (D10) | Cristhian | T00 | ⛔ (Santiago no puede subir cambios sin esto) | Acceso al repositorio | Santiago hace un PR de prueba | En curso: invitación enviada a Shuh3n (2026-11-03), pendiente de aceptar |
| T03 | Configurar el entorno Python (local o Colab) con `requirements.txt` | Ambos | T00 | 🟢 | Entorno funcionando | `pytest` se ejecuta y `import cv2` funciona | ⬜ |
| T04 | Confirmar con la profesora: (a) fecha de entrega, (b) temas pendientes, (c) si la CNN también debe hacerse en MATLAB y (d) si es aceptable entrenar con un dataset extranjero. **Respuesta (2026-10-29):** (c) el proyecto puede hacerse solo en Python, solo en MATLAB o en ambos; (d) se acepta Bremen si el informe dice que no se encontró un dataset público regional de sabotaje, que Bremen se usó para entender y desarrollar el trabajo y que el método funcionaría igual con un dataset regional. (a) y (b): por registrar | Santiago | — | ⛔ (definió D2 y D8) | Respuestas en este plan | Respuestas (c) y (d) escritas | ✅ |
| T05 | **Opcional.** Confirmar versión de MATLAB y toolboxes disponibles (Image Processing, Computer Vision, Deep Learning); solo si se hace la Fase 6 | Santiago | — | 🟢 Opcional | Nota en `docs/matlab.md` | Lista de toolboxes escrita | ⬜ |
| T06 | Justificación regional: verificar y citar fuentes sobre cámaras fuera de servicio o vandalizadas en Colombia y su relevancia en Armenia, Quindío. Incluir en el informe la nota aprobada por la profesora (no se encontró un dataset público regional de sabotaje; Bremen se usó para entender y desarrollar el trabajo; el método se aplica igual a un dataset regional) y citar el dataset de Medellín (D12) como dato regional usado. Fuentes de partida (verificar antes de citar cifras): [Blu Radio, Bucaramanga 808 cámaras](https://www.bluradio.com/regiones/santanderes/videovigilancia-en-bucaramanga-de-808-camaras-sirven-526-rg10), [El Tiempo, Bogotá](https://www.eltiempo.com/bogota/camaras-de-seguridad-fuera-de-servicio-por-robo-vandalismo-y-obras-706006), [Blu Radio, robo de cable](https://www.bluradio.com/regiones/santanderes/robaron-2-000-metros-de-cable-de-camaras-de-seguridad-en-bucaramanga-rg10), [Alcaldía de Armenia](https://armenia.gov.co/alcaldia-de-armenia-rechaza-actos-vandalicos-contra-red-semaforica-y-llama-a-la-ciudadania-a-proteger-la-infraestructura-publica), [Pulzo, Armenia](https://www.pulzo.com/nacion/robo-de-30-millones-en-armenia-camaras-mapas-de-calor-y-reaccion-policial-permiten-captura-PP5150773A) | Cristhian | — | 🟢 (alimenta T101) | `docs/contexto_regional.md` | Cada afirmación tiene su fuente verificada y fecha de consulta | ⬜ |
| T07 | Entorno local de la demo en el portátil que se llevará a la defensa: Python, OpenCV y TensorFlow (CPU); comprobar que la webcam abre (D11) | Santiago | T03 | 🟢 | Instrucciones en `docs/demo.md` | Un script muestra la webcam con los FPS en pantalla | ⬜ |

### Fase 1. Datos

Descargar, conocer y dividir el dataset de Bremen (entrenamiento con la ubicación 1 y prueba de cámara no vista con la ubicación 2) y preparar el dataset regional de Medellín
(prueba cruzada, D12). La división por bloques (D6) evita la fuga de datos; todo lo que sigue se entrena y evalúa
sobre esa división.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T10 | Descargar el dataset de Bremen y registrar las condiciones de uso. Método: WebDAV público, `curl -u "GCsBcswrQf9kBXC:" https://nc.uni-bremen.de/public.php/webdav/<ruta>`; 1,46 GB en 2 ubicaciones × 4 zips (Normal, Blur, Rotated, Noimage) a `data/raw/bremen/zips/`; cada zip trae una carpeta superior (por ejemplo `Location1_Normal`) que se extrae en `data/raw/bremen/` | Cristhian | T00 | ⛔ | `data/README.md` actualizado | Los 8 zips descargados y extraídos; condiciones de uso escritas | ✅ |
| T11 | Inventario: conteo por ubicación y clase, resolución y formato | Cristhian | T10 | ⛔ | `src/data.py`, `results/inventario.csv` | Ubicación 1: 1800/600/600/600 (normal/desenfocada/girada/tapada); ubicación 2: 753/260/167/251; total 5031 | ✅ |
| T12 | Galería: 5 imágenes por clase y cámara | Santiago | T10, T11 | 🟢 | `results/figuras/galeria.png` | La figura muestra todas las clases y cámaras | ⬜ |
| T13 | Analizar nombres y fechas de archivos para detectar casi-duplicados y definir el criterio de división por bloques de tiempo dentro de la ubicación 1 | Cristhian | T11 | ⛔ | Nota en `docs/datos.md` | Criterio de división escrito | ✅ |
| T14 | Crear la división 70/15/15 por bloques de tiempo dentro de la ubicación 1, la función `load_split` y una prueba de no solapamiento | Cristhian | T13 | ⛔ | `data/splits/split.csv`, `src/data.py`, `tests/test_data.py` | `pytest` pasa | ✅ |
| T15 | Elegir imágenes de referencia normales por cámara (para ORB), solo del conjunto train, y documentarlas | Cristhian | T14 | ⛔ (para T33) | Lista en `docs/datos.md` | Cada cámara tiene sus referencias y todas pertenecen a train | ✅ |
| T19 | Definir y documentar el conjunto de prueba de cámara no vista: toda la ubicación 2 (1431 imágenes), nunca usada para entrenar ni calibrar umbrales | Cristhian | T11 | ⛔ (para T43, T64, T71) | `data/splits/camara_no_vista.csv`, nota en `docs/datos.md` | El archivo lista las 1431 imágenes y ninguna aparece en `split.csv` | ✅ |
| T16 | Descargar el dataset de Medellín ([Medellín, Zenodo](https://zenodo.org/records/7935299), DOI 10.5281/zenodo.7935299, `PedestrianDataSet.zip`, 643 MB) a `data/raw/medellin/`; registrar licencia CC BY 4.0 y cita; inventario: conteo, resolución y cámaras si se pueden identificar por nombre de archivo | Santiago | T03 | ⛔ (para T17) | `data/README.md` actualizado, `results/inventario_medellin.csv` | 6324 imágenes JPG de 1280×720; licencia y cita escritas | ⬜ |
| T17 | Seleccionar un subconjunto de ≈600 imágenes normales repartido entre cámaras, excluyendo si es posible las que tengan personas identificables en primer plano; documentar el criterio. Riesgo: el descriptor ORB necesita referencias de la **misma** cámara; si no se puede derivar la cámara del nombre de archivo, agrupar por similitud visual o evaluar la rotación en Medellín con descriptores sin referencia (orientación de Hough) | Santiago | T16 | ⛔ (para T18) | Lista y criterio en `docs/datos.md` | ≈600 imágenes repartidas entre cámaras, con el criterio escrito | ⬜ |
| T18 | Generar las 3 clases saboteadas sobre el subconjunto (desenfoque gaussiano, rotación con recorte, oclusión total y parcial) con semilla fija, más pruebas y script reproducible | Santiago | T17 | ⛔ (para T76) | `src/synthetic_tampering.py`, `tests/test_synthetic_tampering.py`, `data/medellin_sintetico/` (no versionado) | `pytest` pasa y el script regenera el mismo conjunto con la misma semilla | ⬜ |

### Fase 2. Perturbaciones

Funciones que simulan los ataques de luz y ruido (D7). Son la base de la auditoría y de la defensa, y no
dependen de los datos, así que se pueden hacer desde el inicio.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T20 | Implementar brillo, gamma, ruido gaussiano, sal y pimienta y JPEG, con 3 niveles y semilla | Santiago | T03 | ⛔ (para T70) | `src/perturbations.py` | Las pruebas de T21 pasan | ⬜ |
| T21 | Pruebas: conserva forma, `dtype` uint8, rango 0–255 y reproducible con la misma semilla | Santiago | T20 | ⛔ (aceptación de T20) | `tests/test_perturbations.py` | `pytest` pasa | ⬜ |
| T22 | Grilla visual perturbación × nivel | Santiago | T20 | 🟢 | `notebooks/02_perturbaciones.ipynb` | La grilla muestra los 6 tipos de perturbación en 3 niveles | ⬜ |
| T23 | Justificar los niveles elegidos (qué situación real simula cada uno) | Santiago | T20 | 🟢 | Sección en `notebooks/02_perturbaciones.ipynb` | Cada nivel tiene su justificación escrita | ⬜ |

### Fase 3. Métricas

Un solo módulo de métricas para los dos modelos garantiza que se comparen con las mismas fórmulas.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T25 | Matriz de confusión, F1 macro, tasa de falsos positivos (normal predicha como saboteada), ROC binaria con AUC y tiempo por imagen en ms | Cristhian | T03 | ⛔ (para T43, T64, T71) | `src/metrics.py`, `tests/test_metrics.py` | Las pruebas con ejemplos de resultado conocido pasan | ✅ |

### Fase 4. Módulo clásico: descriptores

Calcular los descriptores de D4 y comprobar, con gráficas, cuál separa cada clase. Sin esa evidencia no se
pueden justificar los umbrales.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T30 | Preprocesamiento: escala de grises, tamaño fijo y filtro Gaussiano | Cristhian | T14 | ⛔ | `src/features.py` | Prueba: la salida tiene el tamaño fijo y un solo canal | ✅ |
| T31 | Descriptor de nitidez: varianza del Laplaciano | Cristhian | T30 | ⛔ | `src/features.py` | Prueba: una imagen suavizada da menor valor que la original | ✅ |
| T32 | Descriptores de uniformidad: desviación estándar y entropía del histograma | Cristhian | T30 | ⛔ | `src/features.py` | Prueba: una imagen constante da desviación 0 y entropía 0 | ✅ |
| T33 | Descriptor geométrico con la configuración de la clase 11: ORB (`nfeatures=1000`), `BFMatcher` con `NORM_HAMMING`, `knnMatch` k=2 y razón de Lowe 0,75; puntaje = número de buenas coincidencias contra las referencias; ángulo de rotación por homografía con RANSAC (del artículo) | Cristhian | T15, T30 | ⛔ | `src/features.py` | Prueba: una imagen rotada un ángulo conocido devuelve un ángulo cercano | ✅ |
| T34 | Descriptores complementarios: densidad de bordes Canny, orientación dominante de Hough y HOG (scikit-image: `orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`) con distancia L2 frente a las referencias como señal global de cambio de escena. Cuidado: el notebook de clase calcula la distancia sobre la imagen de visualización (`visualize=True` devuelve `(features, image)`); usar los vectores de características | Cristhian | T30 | 🟢 | `src/features.py` | Prueba sobre una imagen sintética con bordes conocidos | ⬜ |
| T37 | Descriptor de color en HSV: saturación media, dispersión del matiz y proporción de píxeles en rango de tono de piel (mano tapando la cámara); ayuda a separar "tapada" de una escena nocturna oscura (umbralización de color) | Cristhian | T30 | ⛔ (para T35) | `src/features.py` | Prueba: una imagen sintética de tono piel da proporción alta y una gris da saturación cercana a 0 | ✅ |
| T38 | Comparar ORB, SIFT y KAZE como descriptor geométrico en train y val: acierto en "girada", número de buenas coincidencias y tiempo por frame; elegir y justificar (árbol de decisión de la clase 11: ORB para tiempo real, KAZE para robustez). Configuración de clase: KAZE con `NORM_L2` y razón 0,7; SIFT con FLANN (KD-tree, `trees=5`, `checks=50`) y razón 0,7 | Cristhian | T33 | 🟢 | Sección en `notebooks/03_descriptores.ipynb`, `results/clasico/comparacion_descriptores.csv` | Tabla con los tres descriptores y la elección justificada | ⬜ |
| T35 | Calcular los descriptores sobre train, diagramas de caja por clase y anotar qué descriptor separa cada clase | Cristhian | T31, T32, T33, T37 | ⛔ (para T40) | `notebooks/03_descriptores.ipynb` | Las gráficas y la conclusión por clase quedan escritas | ✅ |
| T36 | Ficha exacta de cada descriptor, incluido el de color (fórmula, parámetros, rangos HSV, tamaño de imagen), para el informe y, si se hace, para la réplica opcional en MATLAB | Cristhian | T35 | 🟢 | `docs/descriptores.md` | Cada descriptor se puede reproducir solo con la ficha | ⬜ |

### Fase 5. Módulo clásico: reglas y evaluación

Convertir los descriptores en un clasificador por reglas con umbrales justificados y medirlo con las métricas
de la rúbrica (tiempo por frame y tasa de falsos positivos).

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T40 | Árbol de reglas: tapada → desenfocada → girada → normal (la regla de tapada puede usar el descriptor de color de T37) | Cristhian | T35 | ⛔ | `src/classic.py` | Prueba: clasifica correctamente imágenes de ejemplo de cada clase | ✅ |
| T41 | Calibrar los umbrales con train (percentiles de las normales) y justificarlos con una gráfica | Cristhian | T40 | ⛔ | `results/clasico/umbrales.json` y figura | Cada umbral tiene su gráfica y su justificación | ✅ |
| T42 | Puntaje continuo de sabotaje para poder trazar la ROC | Cristhian | T41 | ⛔ (para T71) | `src/classic.py` | El puntaje es mayor, en promedio, para las saboteadas que para las normales | ✅ |
| T43 | Evaluar en val (ajuste) y luego una sola vez en test y en la cámara no vista: matriz, F1, tasa de falsos positivos y tiempo por frame | Cristhian | T19, T25, T42 | ⛔ | `results/clasico/` | Las métricas de test quedan guardadas | ✅ |
| T44 | Análisis de errores con las imágenes mal clasificadas y su causa | Santiago | T43 | 🟢 | `notebooks/04_clasico.ipynb` | Cada tipo de error tiene ejemplos y causa escrita | ⬜ |

### Fase 6. MATLAB (opcional)

**Opcional, solo si sobra tiempo al final (D8).** La profesora aceptó el proyecto solo en Python (T04). Si se hace,
se replican en MATLAB las perturbaciones y el módulo clásico y se compara con Python. No forma parte de la ruta
crítica ni de los totales de horas (≈ 12 h aparte).

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T50 | Perturbaciones equivalentes a T20 | Santiago | T20 | 🟢 Opcional | `matlab/perturbaciones.m` | Mismos parámetros y niveles que Python | ⬜ |
| T51 | Descriptores: Laplaciano, desviación, entropía, color HSV, Canny y Hough, según la ficha de T36 | Santiago | T36 | 🟢 Opcional | `matlab/descriptores.m` | Valores comparables con Python en imágenes de prueba | ⬜ |
| T52 | ORB y homografía en MATLAB (requiere Computer Vision Toolbox) | Santiago | T05, T36 | 🟢 Opcional | `matlab/geometrico.m` | Ángulo cercano al de Python en imágenes rotadas | ⬜ |
| T53 | Clasificador clásico con los mismos umbrales (leídos de `results/clasico/`) | Santiago | T41, T51, T52 | 🟢 Opcional | `matlab/clasificador_clasico.m` | Clasifica las imágenes de prueba igual que Python | ⬜ |
| T54 | Comparar Python frente a MATLAB en 20 imágenes | Santiago | T53 | 🟢 Opcional | Tabla de diferencias en `results/matlab/` | Diferencias por debajo de una tolerancia definida y explicada | ⬜ |

### Fase 7. Aprendizaje profundo (⏳ ajustable)

Entrenar la CNN (D5). Se empieza ya con estudio propio de Keras y se ajusta cuando se vea el tema en clase (D14). Se usa la misma división (T14) y las mismas métricas (T25)
que el módulo clásico para que la comparación sea justa. Orden: T62 → T66 → T63 → T64 (los hiperparámetros se
eligen con evidencia, no al azar).

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T60 | Cargar los datos desde la división (224×224, normalización) | Santiago | T14 | ⏳ ajustable ⛔ | `src/cnn.py` | Los lotes tienen la forma y el rango esperados | ⬜ |
| T61 | Modelo: MobileNetV2 congelado y cabeza de 4 clases (o la arquitectura vista en clase) | Santiago | T60 | ⏳ ajustable ⛔ | `src/cnn.py` | El resumen del modelo muestra la salida de 4 clases | ⬜ |
| T62 | Entrenar con early stopping y guardar las curvas de loss y accuracy en train y val | Santiago | T61 | ⏳ ajustable ⛔ | `results/cnn/` | Las curvas quedan guardadas | ⬜ |
| T66 | Comparación pequeña de hiperparámetros: tasa de aprendizaje 1e-2, 1e-3 y 1e-4; tamaño de lote 16 y 32; número de capas descongeladas. Comparar las curvas de validación y documentar por qué se elige cada valor | Santiago | T62 | ⏳ ajustable ⛔ | `results/cnn/hiperparametros.csv`, sección en `docs/cnn.md` | Cada valor elegido tiene su curva de validación y su justificación | ⬜ |
| T63 | Diagnosticar sobreajuste y ajustar (aumentación de datos, dropout) | Santiago | T66 | ⏳ ajustable ⛔ | `results/cnn/` | Las curvas de train y val no se separan de forma severa | ⬜ |
| T64 | Evaluar en test y en la cámara no vista con `src/metrics.py` y medir el tiempo por frame | Santiago | T19, T25, T63 | ⏳ ajustable ⛔ (para T71) | `results/cnn/` | Las métricas de test quedan guardadas | ⬜ |
| T65 | Documentar la arquitectura para explicarla en la defensa | Santiago | T63 | ⏳ ajustable 🟢 | `docs/cnn.md` | Capas, parámetros y decisiones explicadas | ⬜ |

### Fase 8. Captura propia y demo

La captura propia en Armenia (D3) sirve para la demo y su calibración, aporta el conjunto de iluminación física
de la auditoría (D7) y una muestra pequeña de sabotaje regional real. La demo en vivo corre en el portátil (D11)
y es la herramienta principal de la defensa.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T80 | Protocolo de captura regional: cámara fija (webcam o celular) en una escena real de Armenia, Quindío (entrada de una casa, edificio o local, con permiso); cómo simular cada sabotaje (tapar con mano, cinta o tela; desenfocar; girar el soporte); conjunto de iluminación física (luz apagada, lámpara, contraluz de ventana) para escenas normales y saboteadas; cumplimiento de la Ley 1581 de 2012: evitar terceros identificables o pedir su consentimiento, y nunca publicar imágenes con personas | Santiago | — | ⛔ (para T81) | `docs/captura.md` | El protocolo cubre las 4 clases, el conjunto de iluminación y las reglas de protección de datos | ⬜ |
| T81 | Capturar ~100 imágenes por clase y el conjunto de iluminación física | Santiago | T80 | ⛔ (para T75, T83, T93) | `data/propios/` (no versionado) | Conteo de ~100 por clase y conjunto de iluminación completo, sin personas identificables | ⬜ |
| T82 | Demo: webcam → clasificación clásica en vivo con FPS en pantalla. Empieza con un paso de calibración que captura 8 frames normales de referencia de la escena en vivo (el método clásico compara ORB contra referencias de la **misma** cámara; las de Bremen no sirven para otra cámara; 8 como en el artículo) | Santiago | T07, T40 | 🟢 | `src/demo.py` | La demo calibra, corre en vivo y muestra la clase y los FPS | ⬜ |
| T83 | Evaluar ambos modelos sobre los datos propios (generalización a otra cámara); ajuste fino de la CNN si falla | Santiago | T43, T64, T81 | ⛔ (para T93) | `results/propios/` | Métricas de ambos modelos sobre los datos propios | ⬜ |
| T84 | Agregar la CNN a la demo | Santiago | T64, T82 | 🟢 | `src/demo.py` | La demo permite elegir el modelo | ⬜ |
| T85 | Teclas en la demo para activar y desactivar los filtros correctivos de T91 (CLAHE, gamma, mediana) y mostrar la predicción antes y después, con los FPS. Se hace primero con el módulo clásico; la CNN se agrega cuando T84 esté lista, sin bloquear el ensayo | Santiago | T82, T91 | ⛔ (para T94) | `src/demo.py` | Cada tecla activa su filtro y la pantalla muestra ambas predicciones y los FPS | ⬜ |

### Fase 9. Auditoría de robustez

Medir cómo se degradan los dos modelos ante luz y ruido, de forma artificial y física (R5, D7), y ante el cambio
de cámaras de Alemania a Colombia (D12). Es el aporte
principal frente al artículo de referencia y la base de la defensa.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T70 | Generar los conjuntos de prueba degradados (test × perturbación × nivel) | Cristhian | T14, T20 | ⛔ | Conjuntos degradados (no versionados) y su script | El número de imágenes coincide con test × 6 × 3 (6 tipos: brillo oscuro y claro por separado, D7) | ✅ |
| T71 | Ejecutar ambos modelos en cada condición: tabla de F1, tasa de falsos positivos y tiempo | Cristhian | T19, T43, T64, T70 | ⛔ | `results/auditoria/tabla.csv` | La tabla cubre todas las condiciones y ambos modelos | En curso: módulo clásico hecho (`docs/auditoria.md`); falta la CNN (PR #11) |
| T72 | Matrices de confusión cruzadas (modelo × condición crítica) | Cristhian | T71 | ⛔ | `results/auditoria/` | Una matriz por modelo y condición crítica | ⬜ |
| T73 | Curvas ROC por condición y curvas de F1 frente a severidad | Cristhian | T71 | ⛔ | `results/auditoria/` | Las curvas quedan guardadas | ⬜ |
| T74 | Notebook de auditoría con la interpretación de los resultados | Cristhian | T72, T73 | ⛔ (para T90) | `notebooks/06_auditoria.ipynb` | Cada condición crítica tiene su interpretación escrita | ⬜ |
| T75 | Auditoría física: ejecutar ambos modelos sobre el conjunto de iluminación física y compararlo con la perturbación artificial equivalente (¿el oscurecimiento artificial predice la luz baja real?) | Cristhian | T71, T81 | ⛔ (para T90) | Sección en `notebooks/06_auditoria.ipynb`, `results/auditoria/fisica.csv` | Tabla física frente a artificial por condición y ambos modelos, con su interpretación | ⬜ |
| T76 | Evaluación regional cruzada: ambos modelos entrenados en Bremen se evalúan en Medellín (sabotaje sintético): matriz de confusión, F1 y tasa de falsos positivos; comparar con el resultado en Bremen y explicar la diferencia | Santiago | T18, T43, T64 | ⛔ (para T90, T101) | `results/regional/`, sección en `notebooks/06_auditoria.ipynb` | Métricas de ambos modelos en Medellín frente a Bremen, con la diferencia explicada | ⬜ |
| T77 | Robustez de ORB, SIFT y KAZE ante las perturbaciones (brillo, gamma, ruido, JPEG): buenas coincidencias frente a severidad. Comprueba lo que dice la clase (ORB pierde precisión con luz y ruido; KAZE es más robusto) y da el argumento de defensa: "con poca luz ORB pierde coincidencias → aplicar CLAHE o cambiar a KAZE" | Santiago | T20, T38 | 🟢 (alimenta T90, T93) | `results/auditoria/descriptores_vs_perturbacion.csv` y figura | Curva por descriptor y perturbación, con la conclusión escrita | ⬜ |

### Fase 10. Defensa

Preparar la respuesta al ataque en vivo (40 % de la nota): para cada ataque, saber explicar matemáticamente por
qué falla el modelo y qué filtro lo corrige, con la mejora medida.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T90 | Diagnóstico de cada fallo: qué descriptor o canal se altera y por qué (Cristhian: clásico; Santiago: CNN) | Ambos | T74, T75, T76 | ⛔ | Notas en `docs/defensa.md` | Cada fallo tiene su causa escrita | ⬜ |
| T91 | Correcciones: CLAHE o ecualización, corrección gamma, mediana, Gaussiano o bilateral | Santiago | T20 | ⛔ (para T92) | `src/corrections.py` | Pruebas: conserva forma y `dtype` | ⬜ |
| T92 | Re-evaluar con las correcciones: tabla antes y después | Cristhian | T71, T91 | ⛔ | `results/defensa/` | Tabla antes/después por perturbación | ⬜ |
| T93 | Guía de defensa: ataque → síntoma → causa matemática → filtro → mejora medida | Ambos | T83, T90, T92 | ⛔ | `docs/defensa.md` | Cubre todas las perturbaciones de D7 | ⬜ |
| T94 | Ensayo cruzado con la demo: uno ataca en vivo y el otro explica y activa el filtro, y viceversa | Ambos | T85, T93 | 🟢 | — | Ambos explican cualquier ataque sin leer | ⬜ |

### Fase 11. Informe y presentación

Consolidar todo en el informe técnico y la presentación final.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T100 | Estructura del informe | Cristhian | T01 | 🟢 | `informe/` | Índice con todas las secciones | ⬜ |
| T101 | Redactar secciones: contexto regional (con la nota aprobada por la profesora), datos, clásico y auditoría (Cristhian); perturbaciones, CNN, dataset regional de Medellín, captura en Armenia, demo y MATLAB si se hizo (Santiago) | Ambos | T06, T76, T100 | 🟢 | `informe/` | Cada sección con sus figuras y métricas | ⬜ |
| T102 | Presentación | Ambos | T101 | 🟢 | `informe/` | Presentación completa | ⬜ |
| T103 | Revisión cruzada final | Ambos | T93, T101, T102 | ⛔ (entrega) | — | Cada uno revisa las secciones del otro | ⬜ |

---

## 8. Temas de la materia pendientes

Cuando se vea un tema nuevo, se revisa esta tabla y se ajustan las tareas y decisiones afectadas.

| Tema | Estado en clase | Tareas afectadas | Qué podría cambiar |
|---|---|---|---|
| Filtros espaciales, bordes (Canny, Sobel), morfología, segmentación (Otsu), extracción de características, texturas (GLCM, LBP, Gabor), momentos de Hu, transformada de Hough | ✅ Visto | T30–T44 (y T50–T54, opcionales) | — |
| Descriptores locales: SIFT, HOG, KAZE, SURF y ORB (clase 11, 2026-11-03) | ✅ Visto | T33, T34, T38, T77 | D4 y D13 actualizados. RANSAC y homografía no se vieron en clase: vienen del artículo |
| Redes neuronales convolucionales y transfer learning | ⏳ Por ver; se empieza con estudio propio (D14) | T60–T66, T71, T76, T83, T84 | D5: arquitectura (el framework ya está decidido: Keras) |
| Métricas formales de evaluación (ROC, F1) | ⏳ Confirmar si se ve en clase | T25 | D9: forma de calcularlas y presentarlas |
| Otros temas que anuncie la profesora | ⏳ | Por definir | Por definir |

---

## 9. Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| La licencia del dataset de Bremen no está escrita en la página de descarga | Usarlo solo con fines académicos, citar el artículo y registrar las condiciones de uso en T10 |
| La CNN no generaliza a la cámara de la demo (lo advierte el propio artículo) | Captura propia (T81) y evaluación con ajuste fino (T83) |
| La captura regional puede registrar personas (Ley 1581 de 2012, protección de datos personales) | Protocolo de T80: evitar terceros identificables o pedir consentimiento, y nunca publicar imágenes con personas; los datos propios no se versionan |
| El sabotaje sintético de Medellín no es idéntico al sabotaje real | Complementarlo con la muestra pequeña de sabotaje real de la captura en Armenia (T81, T83) y explicar la diferencia en T76 |
| Las imágenes de Medellín contienen peatones (Ley 1581 de 2012) | Preferir imágenes sin personas identificables en primer plano (T17); no republicar imágenes con personas; los datos no se versionan |
| Las cámaras de Medellín no se pueden identificar y ORB necesita referencias de la misma cámara | Agrupar por similitud visual o evaluar la rotación con la orientación de Hough, sin referencias (T17) |
| El método clásico falla en la cámara de la demo porque sus referencias ORB son de otra cámara | Calibración con 8 referencias de la escena en vivo al iniciar la demo (T82) |
| Fuga de datos por imágenes casi idénticas | División por bloques de tiempo (T13, T14) con prueba de no solapamiento; la ubicación 2 queda aparte como cámara no vista (T19) |
| Desbalance de clases en la ubicación 2 (solo 167 giradas frente a 753 normales) | Reportar F1 macro y la matriz de confusión por clase, no solo el accuracy (T25) |
| ORB pierde coincidencias con poca luz o ruido (clase 11) | Medirlo (T77) y tener preparada la corrección: CLAHE o cambio a KAZE (T38, T91) |
| Poco tiempo | Tareas pequeñas; T10–T44 no dependen de temas pendientes y se pueden avanzar ya; MATLAB quedó como opcional (D8) |
| Faltan toolboxes de MATLAB (solo si se hace la Fase 6 opcional) | Confirmarlo con T05. Si no hay Computer Vision Toolbox, estimar el giro con la orientación dominante de Hough en lugar de ORB y homografía |
| Una imagen muy oscura se parece a "tapada" | Es un hallazgo esperado: se documenta en la auditoría (T74) y se diagnostica en la defensa (T90) |

---

## 10. Reparto

Cada integrante es dueño de un bloque completo del flujo, para que pueda explicarlo en la defensa. Cristhian se
encarga de la ruta crítica: justificación regional, datos de Bremen, métricas, módulo clásico y auditoría
(artificial y física). Santiago se encarga de perturbaciones, CNN, dataset regional de Medellín, captura en
Armenia y demo, además de la galería de datos (T12) y el análisis de errores del clásico (T44), que se movieron
para equilibrar la carga. Las horas que liberó MATLAB (D8) pasaron al dataset regional (T16–T18, T76). La defensa
y el informe son compartidos. El detalle está en [docs/PLAN_CRISTHIAN.md](PLAN_CRISTHIAN.md) y
[docs/PLAN_SANTIAGO.md](PLAN_SANTIAGO.md).

| Integrante | Tareas propias | Horas propias (estimadas) | Fases principales |
|---|---|---|---|
| Cristhian | 31 (T00 y T10 ya hechas): T00, T02, T06, T10, T11, T13–T15, T19, T25, T30–T38, T40–T43, T70–T75, T92, T100 | ≈ 63 h | Contexto regional, datos, métricas, clásico, auditoría |
| Santiago | 27 (T04 ya hecha): T04, T07, T12, T16–T18, T20–T23, T44, T60–T66, T76, T77, T80–T85, T91 | ≈ 61 h, más el estudio propio de Keras | Perturbaciones, CNN, dataset regional, captura y demo |
| Ambos | 8 compartidas: T01, T03, T90, T93, T94, T101, T102, T103 | ≈ 15 h cada uno | Defensa e informe |
| Opcional (Santiago) | 6: T05, T50–T54 (MATLAB) | ≈ 12 h, fuera de los totales | Solo si sobra tiempo |

**Balance:** la carga queda equilibrada. Cristhian es dueño de la ruta crítica (datos → clásico → auditoría), por
lo que sus tareas deben avanzar primero. Las tareas de aprendizaje profundo de Santiago son más pesadas por tarea
y se empiezan ya con estudio propio (D14). El calendario por días y la verificación de capacidad están en la
sección 6. Santiago también suma el estudio de Keras, que no figura como tarea. Si
la carga se desbalancea, T38 puede pasar a Santiago junto con T77. Las horas son estimaciones (cada tarea nueva ≈ 2–3 h) y se corrigen con el avance real.
