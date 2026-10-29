# Plan general del proyecto: detección de sabotaje en cámaras de vigilancia

**Grupo 7 · Seguridad** · Cristhian Eduardo Osorio Restrepo y Santiago Orozco Zuluaga
**Docente:** Ing. Ana Maria Tamayo Ocampo, Mg · **Fecha de entrega:** _por definir_

Planes personales: [Cristhian](PLAN_CRISTHIAN.md) · [Santiago](PLAN_SANTIAGO.md)

> Este plan es un documento vivo. Las tareas marcadas con ⏳ dependen de temas que aún no se han visto en clase
> y se ajustan cuando la profesora los explique (ver sección 6).

---

## 1. Cómo leer este plan

| Símbolo | Significado |
|---|---|
| ⛔ | **Bloqueante:** otra tarea espera por esta. Hacerla primero. |
| 🟢 | **No bloqueante:** se puede hacer en cualquier momento después de sus dependencias. |
| ⏳ | **Espera un tema de clase:** no se empieza hasta ver el tema (puede ser también bloqueante: ⏳⛔). |
| ✅ | Tarea hecha. |
| ⬜ | Tarea pendiente. |

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
| R1 | Problema regional de clasificación o detección, con temática distinta por grupo | ✅ Definido: clasificar imágenes de cámaras de vigilancia saboteadas (Seguridad). ⬜ Componente regional: no existe un dataset público regional; se entrena con Bremen y se valida con una captura propia en Armenia (Quindío), con justificación regional documentada | T06, T80, T81, T83 |
| R2 | Proceso en Python **y** MATLAB | ⬜ Pendiente. Python: todo el flujo. MATLAB: perturbaciones y módulo clásico; falta confirmar con la profesora si la CNN también debe hacerse en MATLAB | T04, T20–T44, T50–T54 |
| R3 | Módulo 1, clásico (20 %): descriptores matemáticos, transformaciones geométricas, umbralización (incluida la umbralización de color), filtrado lineal y no lineal. Métricas: tiempo de cómputo por frame y tasa de falsos positivos en entornos estables | ⬜ Pendiente. Los temas necesarios ya se vieron en clase. La umbralización de color se cubre con un descriptor HSV (T37) | T30–T44 |
| R4 | Módulo 2, aprendizaje profundo (20 %): CNN propia o transfer learning, con hiperparámetros justificados (no elegidos al azar). Métricas: curvas de loss y accuracy en train y val, sin sobreajuste severo | ⏳ Depende del tema de CNN en clase. Los hiperparámetros se eligen con una comparación pequeña y documentada (T66) | T60–T66 |
| R5 | Módulo 3, auditoría de robustez (20 %): ambos sistemas ante entornos degradados de manera artificial o física (entornos dinámicos reales). Métricas: matrices de confusión cruzadas, curvas ROC y F1-score en condiciones críticas | ⬜ Pendiente. Degradación artificial (perturbaciones, T70–T74) y física (captura con cambios reales de iluminación, T75) | T20–T25, T70–T75, T81 |
| R6 | Defensa ante ataque adversario en vivo (40 %): diagnosticar matemáticamente el fallo y proponer el filtro correctivo exacto | ⬜ Pendiente. Demo local en el portátil (D11) con calibración de referencias y teclas para activar los filtros correctivos en vivo | T07, T82, T85, T90–T94 |
| R7 | Dataset | ✅ Identificado y accesible: ComNets Camera Tampering Images Datasets (Universidad de Bremen), 3600 imágenes, para entrenamiento. ⬜ Captura propia regional en Armenia, Quindío | T10–T15, T80, T81 |

**Brecha principal:** el artículo de referencia compara reglas frente a una CNN, pero no prueba los modelos ante
cambios de iluminación ni ruido, ni en cámaras de otra región. Ese es el aporte de este proyecto: la auditoría de
robustez (artificial y física), la validación con una captura propia en Armenia y la defensa.

---

## 3. Decisiones

| ID | Decisión | Elección | Por qué |
|---|---|---|---|
| D1 | Clases | 4 clases: normal, desenfocada, girada, tapada. Además se reporta la versión binaria (normal frente a saboteada) | Las 4 clases permiten diagnosticar el tipo de sabotaje; la versión binaria permite una curva ROC directa |
| D2 | Dataset principal | ComNets Camera Tampering Images Datasets (Universidad de Bremen), citando el artículo, **para entrenamiento**. El componente regional lo aportan la captura propia en Armenia (D3) y la justificación regional (T06) | Imágenes reales, etiquetadas y de tamaño manejable. No existe un dataset público regional: la búsqueda en datos.gov.co y de cámaras web públicas de Armenia o Salento no dio resultados |
| D3 | Datos propios (componente central) | Captura regional con webcam o celular fijo en una escena real de Armenia, Quindío (~100 imágenes por clase), más un conjunto con cambios físicos de iluminación (D7). Se respeta la Ley 1581 de 2012 de protección de datos personales | Es el componente regional del proyecto. Además, la demo en vivo usará otra cámara y el artículo advierte que la CNN falla si la cámara cambia: sirve para la demo, para medir generalización y para la auditoría física |
| D4 | Módulo clásico | Árbol de reglas con umbrales calibrados y justificados. Descriptores: varianza del Laplaciano (nitidez), desviación estándar y entropía del histograma (tapada), coincidencias ORB contra referencias y ángulo por homografía (girada); descriptor de color en HSV: saturación media, dispersión del matiz y proporción de píxeles en rango de tono de piel (mano tapando la cámara), que ayuda a separar "tapada" de una escena nocturna oscura; complementarios: densidad de bordes Canny y orientación dominante de Hough. Preprocesamiento: escala de grises, tamaño fijo, filtro Gaussiano (lineal) y mediana (no lineal) | Sigue el método validado del artículo, usa los temas vistos en clase y cubre lo que pide la rúbrica: descriptores matemáticos, transformaciones geométricas, umbralización (incluida la **umbralización de color**, que el resto de descriptores en gris no cubre) y filtrado lineal y no lineal |
| D5 | Módulo profundo | ✅ Framework decidido: **TensorFlow/Keras** (2026-10-29). Transfer learning con MobileNetV2 y una cabeza de 4 clases; la arquitectura se ajusta a lo que se vea en clase (⏳) | Keras es simple de escribir y de explicar, entrega las curvas de loss y accuracy, corre con GPU gratis en Colab y es el framework del artículo de referencia. El transfer learning converge rápido con pocos datos y reduce el sobreajuste |
| D6 | División de datos | train/val/test 70/15/15 por bloques de tiempo o ubicación de cámara, no aleatoria imagen por imagen. El archivo de división se versiona en `data/splits/` | Los frames consecutivos son casi idénticos: una división aleatoria filtra información entre conjuntos e infla las métricas |
| D7 | Perturbaciones de la auditoría | Brillo (oscurecer y aclarar), corrección gamma, ruido gaussiano, sal y pimienta y compresión JPEG; 3 niveles cada una; semilla fija. **El desenfoque no se usa como perturbación.** Además, degradación **física**: captura propia con cambios reales de iluminación (luz apagada, lámpara, contraluz de ventana) para escenas normales y saboteadas | Luz y ruido son los ataques probables en la defensa. El desenfoque es una clase ("desenfocada"): una imagen normal desenfocada cambiaría de etiqueta legítimamente. Nota: una imagen muy oscura puede parecerse a "tapada"; esa confusión es un hallazgo esperado que se debe diagnosticar. La guía admite degradación "artificial o física" y el título habla de "entornos dinámicos reales": la parte física comprueba si las perturbaciones artificiales predicen lo que pasa con luz real |
| D8 | Python y MATLAB | Python: todo el flujo. MATLAB: réplica de perturbaciones y módulo clásico (CNN en MATLAB opcional). **El alcance final depende de T04** (confirmar con la profesora si la CNN también debe hacerse en MATLAB) | Cumple R2 sin duplicar el entrenamiento de la red |
| D9 | Métricas | Las de la rúbrica: tiempo por frame, tasa de falsos positivos, curvas de loss y accuracy, matrices de confusión cruzadas, ROC y F1. El módulo clásico debe producir un **puntaje continuo de sabotaje** para poder trazar su ROC | Son exactamente las exigidas; un clasificador por reglas solo da etiquetas, y la ROC necesita un puntaje |
| D10 | Flujo de trabajo con Git | Una rama por tarea (por ejemplo `feat/T14-division-datos`), Pull Request a `main` y revisión del otro integrante; commits convencionales; `main` siempre ejecutable; nunca se suben datos ni modelos | Permite trabajar en paralelo sin romper el trabajo del otro y deja trazabilidad de cada tarea |
| D11 | Entorno de la demo en vivo | La demo corre **localmente en el portátil** con OpenCV (`VideoCapture`) y TensorFlow en CPU; Colab se usa solo para entrenar | Colab no puede acceder a la webcam en tiempo real. La defensa (40 %) depende de que la demo funcione sin conexión ni retrasos |

---

## 4. Ruta crítica

La ruta crítica es la cadena de tareas bloqueantes que define la duración mínima del proyecto. Un retraso en
cualquiera de ellas retrasa la entrega; por eso se priorizan sobre las tareas 🟢.

```
Cadena principal:
T10 → T11 → T13 → T14 → T30 … T37 → T40 … T43 → T70 … T75 → T90 … T94

Entradas a la cadena:
T25 (métricas) ──────────────────► T43, T64, T71
T20 (perturbaciones) ────────────► T70
T60 … T66 ⏳ (CNN) ──────────────► T71
T80 → T81 (captura regional) ────► T75, T83
T82 → T85 (demo con filtros) ────► T94
```

- **T25** (métricas) alimenta la evaluación del clásico (T43), de la CNN (T64) y la auditoría (T71).
- **T20** (perturbaciones) alimenta los conjuntos degradados (T70).
- **T80 → T81** (captura regional) alimenta la auditoría física (T75) y la prueba de generalización (T83).
- **T85** (filtros en la demo) es necesaria para el ensayo de la defensa (T94).
- **T60 … T66** esperan el tema de CNN; mientras tanto se avanza todo lo demás.

---

## 5. Tareas por fase

### Fase 0. Organización

Dejar listo el repositorio, el entorno y los acuerdos de trabajo. Sin esto, ninguno de los dos puede subir
cambios de forma ordenada.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T00 | Crear repositorio, estructura de carpetas y plan inicial | Cristhian | — | ⛔ | Repositorio en GitHub | El repositorio existe en GitHub | ✅ |
| T01 | Leer y acordar este plan entre los dos | Ambos | T00 | ⛔ | Comentario o acuerdo en el PR del plan | Ambos aprueban | ⬜ |
| T02 | Agregar a Santiago como colaborador y acordar el flujo de ramas (D10) | Cristhian | T00 | ⛔ (Santiago no puede subir cambios sin esto) | Acceso al repositorio | Santiago hace un PR de prueba | ⬜ |
| T03 | Configurar el entorno Python (local o Colab) con `requirements.txt` | Ambos | T00 | 🟢 | Entorno funcionando | `pytest` se ejecuta y `import cv2` funciona | ⬜ |
| T04 | Confirmar con la profesora: (a) fecha de entrega, (b) temas pendientes, (c) si la CNN también debe hacerse en MATLAB y (d) si es aceptable entrenar con un dataset extranjero y validar con una captura propia regional | Santiago | — | ⛔ (alcance final de la Fase 6, D8; confirma D2 y D3) | Nota en este plan | Las cuatro respuestas quedan escritas | ⬜ |
| T05 | Confirmar versión de MATLAB y toolboxes disponibles (Image Processing, Computer Vision, Deep Learning) | Santiago | — | ⛔ (para T52) | Nota en `docs/matlab.md` | Lista de toolboxes escrita | ⬜ |
| T06 | Justificación regional: verificar y citar fuentes sobre cámaras fuera de servicio o vandalizadas en Colombia y su relevancia en Armenia, Quindío. Fuentes de partida (verificar antes de citar cifras): [Blu Radio, Bucaramanga 808 cámaras](https://www.bluradio.com/regiones/santanderes/videovigilancia-en-bucaramanga-de-808-camaras-sirven-526-rg10), [El Tiempo, Bogotá](https://www.eltiempo.com/bogota/camaras-de-seguridad-fuera-de-servicio-por-robo-vandalismo-y-obras-706006), [Blu Radio, robo de cable](https://www.bluradio.com/regiones/santanderes/robaron-2-000-metros-de-cable-de-camaras-de-seguridad-en-bucaramanga-rg10), [Alcaldía de Armenia](https://armenia.gov.co/alcaldia-de-armenia-rechaza-actos-vandalicos-contra-red-semaforica-y-llama-a-la-ciudadania-a-proteger-la-infraestructura-publica), [Pulzo, Armenia](https://www.pulzo.com/nacion/robo-de-30-millones-en-armenia-camaras-mapas-de-calor-y-reaccion-policial-permiten-captura-PP5150773A) | Cristhian | — | 🟢 (alimenta T101) | `docs/contexto_regional.md` | Cada afirmación tiene su fuente verificada y fecha de consulta | ⬜ |
| T07 | Entorno local de la demo en el portátil que se llevará a la defensa: Python, OpenCV y TensorFlow (CPU); comprobar que la webcam abre (D11) | Santiago | T03 | 🟢 | Instrucciones en `docs/demo.md` | Un script muestra la webcam con los FPS en pantalla | ⬜ |

### Fase 1. Datos

Descargar, conocer y dividir el dataset. La división por bloques (D6) evita la fuga de datos; todo lo que
sigue se entrena y evalúa sobre esa división.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T10 | Descargar el dataset a `data/raw/` y registrar las condiciones de uso | Cristhian | T00 | ⛔ | `data/README.md` actualizado | Las imágenes están en `data/raw/` y las condiciones de uso quedan escritas | ⬜ |
| T11 | Inventario: conteo por clase y cámara, resolución y formato | Cristhian | T10 | ⛔ | `src/data.py`, `results/inventario.csv` | Conteos 1800/600/600/600 | ⬜ |
| T12 | Galería: 5 imágenes por clase y cámara | Santiago | T10, T11 | 🟢 | `results/figuras/galeria.png` | La figura muestra todas las clases y cámaras | ⬜ |
| T13 | Analizar nombres y fechas de archivos para detectar casi-duplicados y definir el criterio de división | Cristhian | T11 | ⛔ | Nota en `docs/datos.md` | Criterio de división escrito | ⬜ |
| T14 | Crear la división 70/15/15 por bloques, la función `load_split` y una prueba de no solapamiento | Cristhian | T13 | ⛔ | `data/splits/split.csv`, `src/data.py`, `tests/test_data.py` | `pytest` pasa | ⬜ |
| T15 | Elegir imágenes de referencia normales por cámara (para ORB), solo del conjunto train, y documentarlas | Cristhian | T14 | ⛔ (para T33) | Lista en `docs/datos.md` | Cada cámara tiene sus referencias y todas pertenecen a train | ⬜ |

### Fase 2. Perturbaciones

Funciones que simulan los ataques de luz y ruido (D7). Son la base de la auditoría y de la defensa, y no
dependen de los datos, así que se pueden hacer desde el inicio.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T20 | Implementar brillo, gamma, ruido gaussiano, sal y pimienta y JPEG, con 3 niveles y semilla | Santiago | T03 | ⛔ (para T50, T70) | `src/perturbations.py` | Las pruebas de T21 pasan | ⬜ |
| T21 | Pruebas: conserva forma, `dtype` uint8, rango 0–255 y reproducible con la misma semilla | Santiago | T20 | ⛔ (aceptación de T20) | `tests/test_perturbations.py` | `pytest` pasa | ⬜ |
| T22 | Grilla visual perturbación × nivel | Santiago | T20 | 🟢 | `notebooks/02_perturbaciones.ipynb` | La grilla muestra las 5 perturbaciones en 3 niveles | ⬜ |
| T23 | Justificar los niveles elegidos (qué situación real simula cada uno) | Santiago | T20 | 🟢 | Sección en `notebooks/02_perturbaciones.ipynb` | Cada nivel tiene su justificación escrita | ⬜ |

### Fase 3. Métricas

Un solo módulo de métricas para los dos modelos garantiza que se comparen con las mismas fórmulas.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T25 | Matriz de confusión, F1 macro, tasa de falsos positivos (normal predicha como saboteada), ROC binaria con AUC y tiempo por imagen en ms | Cristhian | T03 | ⛔ (para T43, T64, T71) | `src/metrics.py`, `tests/test_metrics.py` | Las pruebas con ejemplos de resultado conocido pasan | ⬜ |

### Fase 4. Módulo clásico: descriptores

Calcular los descriptores de D4 y comprobar, con gráficas, cuál separa cada clase. Sin esa evidencia no se
pueden justificar los umbrales.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T30 | Preprocesamiento: escala de grises, tamaño fijo y filtro Gaussiano | Cristhian | T14 | ⛔ | `src/features.py` | Prueba: la salida tiene el tamaño fijo y un solo canal | ⬜ |
| T31 | Descriptor de nitidez: varianza del Laplaciano | Cristhian | T30 | ⛔ | `src/features.py` | Prueba: una imagen suavizada da menor valor que la original | ⬜ |
| T32 | Descriptores de uniformidad: desviación estándar y entropía del histograma | Cristhian | T30 | ⛔ | `src/features.py` | Prueba: una imagen constante da desviación 0 y entropía 0 | ⬜ |
| T33 | Descriptor geométrico: coincidencias ORB contra las referencias y ángulo por homografía | Cristhian | T15, T30 | ⛔ | `src/features.py` | Prueba: una imagen rotada un ángulo conocido devuelve un ángulo cercano | ⬜ |
| T34 | Descriptores complementarios: densidad de bordes Canny y orientación dominante de Hough | Cristhian | T30 | 🟢 | `src/features.py` | Prueba sobre una imagen sintética con bordes conocidos | ⬜ |
| T37 | Descriptor de color en HSV: saturación media, dispersión del matiz y proporción de píxeles en rango de tono de piel (mano tapando la cámara); ayuda a separar "tapada" de una escena nocturna oscura (umbralización de color) | Cristhian | T30 | ⛔ (para T35) | `src/features.py` | Prueba: una imagen sintética de tono piel da proporción alta y una gris da saturación cercana a 0 | ⬜ |
| T35 | Calcular los descriptores sobre train, diagramas de caja por clase y anotar qué descriptor separa cada clase | Cristhian | T31, T32, T33, T37 | ⛔ (para T40) | `notebooks/03_descriptores.ipynb` | Las gráficas y la conclusión por clase quedan escritas | ⬜ |
| T36 | Ficha exacta de cada descriptor, incluido el de color (fórmula, parámetros, rangos HSV, tamaño de imagen) para replicarlo en MATLAB | Cristhian | T35 | ⛔ (para T51–T53) | `docs/descriptores.md` | Santiago puede replicar cada descriptor solo con la ficha | ⬜ |

### Fase 5. Módulo clásico: reglas y evaluación

Convertir los descriptores en un clasificador por reglas con umbrales justificados y medirlo con las métricas
de la rúbrica (tiempo por frame y tasa de falsos positivos).

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T40 | Árbol de reglas: tapada → desenfocada → girada → normal (la regla de tapada puede usar el descriptor de color de T37) | Cristhian | T35 | ⛔ | `src/classic.py` | Prueba: clasifica correctamente imágenes de ejemplo de cada clase | ⬜ |
| T41 | Calibrar los umbrales con train (percentiles de las normales) y justificarlos con una gráfica | Cristhian | T40 | ⛔ (para T53) | `results/clasico/umbrales.json` y figura | Cada umbral tiene su gráfica y su justificación | ⬜ |
| T42 | Puntaje continuo de sabotaje para poder trazar la ROC | Cristhian | T41 | ⛔ (para T71) | `src/classic.py` | El puntaje es mayor, en promedio, para las saboteadas que para las normales | ⬜ |
| T43 | Evaluar en val (ajuste) y luego una sola vez en test: matriz, F1, tasa de falsos positivos y tiempo por frame | Cristhian | T25, T42 | ⛔ | `results/clasico/` | Las métricas de test quedan guardadas | ⬜ |
| T44 | Análisis de errores con las imágenes mal clasificadas y su causa | Santiago | T43 | 🟢 | `notebooks/04_clasico.ipynb` | Cada tipo de error tiene ejemplos y causa escrita | ⬜ |

### Fase 6. MATLAB

Replicar en MATLAB las perturbaciones y el módulo clásico (R2, D8) y comprobar que dan los mismos resultados
que Python. El alcance final (si se agrega la CNN en MATLAB) se define con la respuesta de T04.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T50 | Perturbaciones equivalentes a T20 | Santiago | T20 | 🟢 | `matlab/perturbaciones.m` | Mismos parámetros y niveles que Python | ⬜ |
| T51 | Descriptores: Laplaciano, desviación, entropía, color HSV, Canny y Hough, según la ficha de T36 | Santiago | T36 | 🟢 | `matlab/descriptores.m` | Valores comparables con Python en imágenes de prueba | ⬜ |
| T52 | ORB y homografía en MATLAB (requiere Computer Vision Toolbox) | Santiago | T05, T36 | 🟢 | `matlab/geometrico.m` | Ángulo cercano al de Python en imágenes rotadas | ⬜ |
| T53 | Clasificador clásico con los mismos umbrales (leídos de `results/clasico/`) | Santiago | T41, T51, T52 | 🟢 | `matlab/clasificador_clasico.m` | Clasifica las imágenes de prueba igual que Python | ⬜ |
| T54 | Comparar Python frente a MATLAB en 20 imágenes | Santiago | T53 | 🟢 | Tabla de diferencias en `results/matlab/` | Diferencias por debajo de una tolerancia definida y explicada | ⬜ |

### Fase 7. Aprendizaje profundo ⏳

Entrenar la CNN (D5) cuando se vea el tema en clase. Se usa la misma división (T14) y las mismas métricas (T25)
que el módulo clásico para que la comparación sea justa. Orden: T62 → T66 → T63 → T64 (los hiperparámetros se
eligen con evidencia, no al azar).

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T60 | Cargar los datos desde la división (224×224, normalización) | Santiago | T14, tema CNN | ⏳⛔ | `src/cnn.py` | Los lotes tienen la forma y el rango esperados | ⬜ |
| T61 | Modelo: MobileNetV2 congelado y cabeza de 4 clases (o la arquitectura vista en clase) | Santiago | T60 | ⏳⛔ | `src/cnn.py` | El resumen del modelo muestra la salida de 4 clases | ⬜ |
| T62 | Entrenar con early stopping y guardar las curvas de loss y accuracy en train y val | Santiago | T61 | ⏳⛔ | `results/cnn/` | Las curvas quedan guardadas | ⬜ |
| T66 | Comparación pequeña de hiperparámetros: tasa de aprendizaje 1e-2, 1e-3 y 1e-4; tamaño de lote 16 y 32; número de capas descongeladas. Comparar las curvas de validación y documentar por qué se elige cada valor | Santiago | T62 | ⏳⛔ | `results/cnn/hiperparametros.csv`, sección en `docs/cnn.md` | Cada valor elegido tiene su curva de validación y su justificación | ⬜ |
| T63 | Diagnosticar sobreajuste y ajustar (aumentación de datos, dropout) | Santiago | T66 | ⏳⛔ | `results/cnn/` | Las curvas de train y val no se separan de forma severa | ⬜ |
| T64 | Evaluar en test con `src/metrics.py` y medir el tiempo por frame | Santiago | T25, T63 | ⏳⛔ (para T71) | `results/cnn/` | Las métricas de test quedan guardadas | ⬜ |
| T65 | Documentar la arquitectura para explicarla en la defensa | Santiago | T63 | ⏳🟢 | `docs/cnn.md` | Capas, parámetros y decisiones explicadas | ⬜ |

### Fase 8. Captura propia y demo

La captura regional en Armenia es el componente regional del proyecto (D3): mide la generalización a otra
cámara y aporta el conjunto de iluminación física de la auditoría (D7). La demo en vivo corre en el portátil (D11)
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

Medir cómo se degradan los dos modelos ante luz y ruido, de forma artificial y física (R5, D7). Es el aporte
principal frente al artículo de referencia y la base de la defensa.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T70 | Generar los conjuntos de prueba degradados (test × perturbación × nivel) | Cristhian | T14, T20 | ⛔ | Conjuntos degradados (no versionados) y su script | El número de imágenes coincide con test × 5 × 3 | ⬜ |
| T71 | Ejecutar ambos modelos en cada condición: tabla de F1, tasa de falsos positivos y tiempo | Cristhian | T43, T64, T70 | ⛔ | `results/auditoria/tabla.csv` | La tabla cubre todas las condiciones y ambos modelos | ⬜ |
| T72 | Matrices de confusión cruzadas (modelo × condición crítica) | Cristhian | T71 | ⛔ | `results/auditoria/` | Una matriz por modelo y condición crítica | ⬜ |
| T73 | Curvas ROC por condición y curvas de F1 frente a severidad | Cristhian | T71 | ⛔ | `results/auditoria/` | Las curvas quedan guardadas | ⬜ |
| T74 | Notebook de auditoría con la interpretación de los resultados | Cristhian | T72, T73 | ⛔ (para T90) | `notebooks/06_auditoria.ipynb` | Cada condición crítica tiene su interpretación escrita | ⬜ |
| T75 | Auditoría física: ejecutar ambos modelos sobre el conjunto de iluminación física y compararlo con la perturbación artificial equivalente (¿el oscurecimiento artificial predice la luz baja real?) | Cristhian | T71, T81 | ⛔ (para T90) | Sección en `notebooks/06_auditoria.ipynb`, `results/auditoria/fisica.csv` | Tabla física frente a artificial por condición y ambos modelos, con su interpretación | ⬜ |

### Fase 10. Defensa

Preparar la respuesta al ataque en vivo (40 % de la nota): para cada ataque, saber explicar matemáticamente por
qué falla el modelo y qué filtro lo corrige, con la mejora medida.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T90 | Diagnóstico de cada fallo: qué descriptor o canal se altera y por qué (Cristhian: clásico; Santiago: CNN) | Ambos | T74, T75 | ⛔ | Notas en `docs/defensa.md` | Cada fallo tiene su causa escrita | ⬜ |
| T91 | Correcciones: CLAHE o ecualización, corrección gamma, mediana, Gaussiano o bilateral | Santiago | T20 | ⛔ (para T92) | `src/corrections.py` | Pruebas: conserva forma y `dtype` | ⬜ |
| T92 | Re-evaluar con las correcciones: tabla antes y después | Cristhian | T71, T91 | ⛔ | `results/defensa/` | Tabla antes/después por perturbación | ⬜ |
| T93 | Guía de defensa: ataque → síntoma → causa matemática → filtro → mejora medida | Ambos | T83, T90, T92 | ⛔ | `docs/defensa.md` | Cubre todas las perturbaciones de D7 | ⬜ |
| T94 | Ensayo cruzado con la demo: uno ataca en vivo y el otro explica y activa el filtro, y viceversa | Ambos | T85, T93 | 🟢 | — | Ambos explican cualquier ataque sin leer | ⬜ |

### Fase 11. Informe y presentación

Consolidar todo en el informe técnico y la presentación final.

| ID | Tarea | Responsable | Depende de | Tipo | Entregable | Verificación | Estado |
|---|---|---|---|---|---|---|---|
| T100 | Estructura del informe | Cristhian | T01 | 🟢 | `informe/` | Índice con todas las secciones | ⬜ |
| T101 | Redactar secciones: contexto regional, datos, clásico y auditoría (Cristhian); perturbaciones, CNN, captura regional, demo y MATLAB (Santiago) | Ambos | T06, T100 | 🟢 | `informe/` | Cada sección con sus figuras y métricas | ⬜ |
| T102 | Presentación | Ambos | T101 | 🟢 | `informe/` | Presentación completa | ⬜ |
| T103 | Revisión cruzada final | Ambos | T93, T101, T102 | ⛔ (entrega) | — | Cada uno revisa las secciones del otro | ⬜ |

---

## 6. Temas de la materia pendientes

Cuando se vea un tema nuevo, se revisa esta tabla y se ajustan las tareas y decisiones afectadas.

| Tema | Estado en clase | Tareas afectadas | Qué podría cambiar |
|---|---|---|---|
| Filtros espaciales, bordes (Canny, Sobel), morfología, segmentación (Otsu), extracción de características, texturas (GLCM, LBP, Gabor), momentos de Hu, transformada de Hough | ✅ Visto | T30–T44, T50–T54 | — |
| Redes neuronales convolucionales y transfer learning | ⏳ Por ver | T60–T66, T71, T83, T84 | D5: arquitectura (el framework ya está decidido: Keras) |
| Métricas formales de evaluación (ROC, F1) | ⏳ Confirmar si se ve en clase | T25 | D9: forma de calcularlas y presentarlas |
| Otros temas que anuncie la profesora | ⏳ | Por definir | Por definir |

---

## 7. Riesgos y mitigación

| Riesgo | Mitigación |
|---|---|
| La licencia del dataset no está escrita en la página de descarga | Usarlo solo con fines académicos, citar el artículo y registrar las condiciones de uso en T10 |
| La CNN no generaliza a la cámara de la demo (lo advierte el propio artículo) | Captura propia (T81) y evaluación con ajuste fino (T83) |
| La captura regional puede registrar personas (Ley 1581 de 2012, protección de datos personales) | Protocolo de T80: evitar terceros identificables o pedir consentimiento, y nunca publicar imágenes con personas; los datos propios no se versionan |
| La profesora no acepta entrenar con un dataset extranjero, o exige la CNN también en MATLAB | Preguntarlo pronto (T04) y ajustar D2, D3 y D8 antes de avanzar la Fase 6 |
| El método clásico falla en la cámara de la demo porque sus referencias ORB son de otra cámara | Calibración con 8 referencias de la escena en vivo al iniciar la demo (T82) |
| Fuga de datos por imágenes casi idénticas | División por bloques (T13, T14) con prueba de no solapamiento |
| Poco tiempo | Tareas pequeñas; T10–T54 no dependen de temas pendientes y se pueden avanzar ya |
| Faltan toolboxes de MATLAB | Confirmarlo pronto (T05). Si no hay Computer Vision Toolbox, estimar el giro con la orientación dominante de Hough en lugar de ORB y homografía |
| Una imagen muy oscura se parece a "tapada" | Es un hallazgo esperado: se documenta en la auditoría (T74) y se diagnostica en la defensa (T90) |

---

## 8. Reparto

Cada integrante es dueño de un bloque completo del flujo, para que pueda explicarlo en la defensa. Cristhian se
encarga de la justificación regional, datos, métricas, módulo clásico y auditoría (artificial y física);
Santiago, de perturbaciones, MATLAB, CNN, captura regional y demo, además de la galería de datos (T12) y el
análisis de errores del clásico (T44). La defensa y el informe son compartidos. El detalle de cada uno está en
[docs/PLAN_CRISTHIAN.md](PLAN_CRISTHIAN.md) y [docs/PLAN_SANTIAGO.md](PLAN_SANTIAGO.md).

| Integrante | Tareas propias | Horas propias (estimadas) | Fases principales |
|---|---|---|---|
| Cristhian | 29 (T00 ya hecha): T00, T02, T06, T10, T11, T13–T15, T25, T30–T37, T40–T43, T70–T75, T92, T100 | ≈ 58 h | Contexto regional, datos, métricas, clásico, auditoría |
| Santiago | 28: T04, T05, T07, T12, T20–T23, T44, T50–T54, T60–T66, T80–T85, T91 | ≈ 60 h | Perturbaciones, MATLAB, CNN, captura regional y demo |
| Ambos | 8 compartidas: T01, T03, T90, T93, T94, T101, T102, T103 | ≈ 15 h cada uno | Defensa e informe |

**Balance:** la carga queda aproximadamente equilibrada. Cristhian es dueño de la ruta crítica (datos →
clásico → auditoría), por lo que sus tareas deben avanzar primero. Las tareas de aprendizaje profundo de
Santiago son más pesadas por tarea y esperan el tema de CNN en clase; mientras tanto, Santiago adelanta
perturbaciones, MATLAB, captura regional y demo. Las horas son estimaciones (cada tarea nueva ≈ 2–3 h) y se
corrigen con el avance real.
