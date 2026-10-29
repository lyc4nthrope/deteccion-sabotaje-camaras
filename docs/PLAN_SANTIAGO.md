# Plan personal: Santiago Orozco Zuluaga

Derivado del [plan general](PLAN.md). Los IDs, dependencias y tipos son los mismos; si algo cambia, se cambia
primero en el plan general.

## Tu rol

Eres dueño de las perturbaciones, la réplica en MATLAB, el modelo de aprendizaje profundo, la captura regional en
Armenia (componente regional del proyecto) y la demo en vivo, además de la galería de datos (T12) y el análisis de errores del clásico (T44). Las perturbaciones
(T20) y las correcciones (T91) son la base de la auditoría y de la defensa; la CNN y la demo son lo que se verá en
vivo frente a la profesora.

Símbolos: ⛔ bloqueante · 🟢 no bloqueante · ⏳ espera un tema de clase. Cada tarea termina con verificación,
commit y Pull Request revisado por Cristhian. Para subir cambios necesitas que Cristhian termine T02.

## Puedes empezar ya

- T01 Leer y acordar el plan (con Cristhian).
- T03 Configurar el entorno Python.
- T04 Confirmar con la profesora fecha, temas pendientes, alcance de MATLAB y uso del dataset extranjero (urgente).
- T05 Confirmar versión de MATLAB y toolboxes.
- T20 Perturbaciones (solo depende de T03; no necesita los datos).
- T07 Entorno local de la demo (después de T03).
- T80 Protocolo de captura regional.

## Lista de tareas

### Fase 0. Organización

- [ ] **T01 ⛔** Leer y acordar el plan con Cristhian.
  - **Qué hacer:** leer el plan general y los dos planes personales; anotar dudas en el PR del plan.
  - **Cómo verificar:** ambos aprueban el PR.
- [ ] **T03 🟢** Configurar el entorno Python (local o Colab) con `requirements.txt`.
  - **Cómo verificar:** `pytest` se ejecuta y `import cv2` funciona.
- [ ] **T04 ⛔** Confirmar con la profesora cuatro puntos.
  - **Qué hacer:** preguntar (a) fecha de entrega, (b) cuándo se ven CNN y métricas formales, (c) si la CNN también debe hacerse en MATLAB y (d) si es aceptable entrenar con un dataset extranjero y validar con una captura propia regional.
  - **Entregable:** respuestas escritas en `docs/PLAN.md` (fecha en el encabezado).
  - **Quién te espera:** el alcance final de la Fase 6 (D8) y la confirmación de D2 y D3; avisa a Cristhian con las respuestas.
- [ ] **T05 ⛔** Confirmar versión de MATLAB y toolboxes (Image Processing, Computer Vision, Deep Learning).
  - **Qué hacer:** ejecutar `ver` en MATLAB y anotar el resultado.
  - **Entregable:** `docs/matlab.md`.
  - **Cómo verificar:** lista de toolboxes escrita.
  - **Quién te espera:** tú mismo (T52). Si no hay Computer Vision Toolbox, avisa a Cristhian para usar Hough en lugar de ORB en MATLAB.
- [ ] **T07 🟢** Entorno local de la demo en el portátil que se llevará a la defensa (D11).
  - **Qué hacer:** instalar Python, OpenCV y TensorFlow (CPU) en el portátil; la demo no corre en Colab porque Colab no accede a la webcam en tiempo real.
  - **Entregable:** instrucciones en `docs/demo.md`.
  - **Cómo verificar:** un script muestra la webcam con los FPS en pantalla.

### Fase 1. Datos (tu parte)

- [ ] **T12 🟢** Galería: 5 imágenes por clase y cámara.
  - **Esperas a:** Cristhian (T10 descarga y T11 inventario).
  - **Qué hacer:** leer `results/inventario.csv` y armar una figura con 5 imágenes por clase y cámara.
  - **Entregable:** `results/figuras/galeria.png`.
  - **Cómo verificar:** la figura muestra todas las clases y cámaras.

### Fase 2. Perturbaciones

- [ ] **T20 ⛔** `src/perturbations.py`.
  - **Qué hacer:** funciones de brillo (oscurecer y aclarar), corrección gamma, ruido gaussiano, sal y pimienta y compresión JPEG; 3 niveles cada una; parámetro de semilla. **No incluir desenfoque** (es una clase, ver D7).
  - **Entregable:** `src/perturbations.py`.
  - **Cómo verificar:** las pruebas de T21 pasan.
  - **Quién te espera:** Cristhian (T70, conjuntos degradados); tú mismo (T50, T91).
- [ ] **T21 ⛔** `tests/test_perturbations.py`.
  - **Qué hacer:** probar que cada perturbación conserva forma, `dtype` uint8, rango 0–255 y es reproducible con la misma semilla.
  - **Cómo verificar:** `pytest` pasa.
- [ ] **T22 🟢** `notebooks/02_perturbaciones.ipynb`: grilla perturbación × nivel.
- [ ] **T23 🟢** Justificar los niveles: qué situación real simula cada uno (anochecer, contraluz, sensor ruidoso, transmisión comprimida…).

### Fase 5. Clásico (tu parte)

- [ ] **T44 🟢** `notebooks/04_clasico.ipynb`: análisis de errores del módulo clásico.
  - **Esperas a:** Cristhian (T43, evaluación del clásico).
  - **Qué hacer:** mostrar las imágenes mal clasificadas, agruparlas por tipo de error y escribir la causa (qué descriptor falló).
  - **Cómo verificar:** cada tipo de error tiene ejemplos y causa escrita.

### Fase 6. MATLAB

- [ ] **T50 🟢** `matlab/perturbaciones.m`, equivalente a T20 (mismos parámetros y niveles).
- [ ] **T51 🟢** `matlab/descriptores.m`: Laplaciano, desviación, entropía, color HSV, Canny y Hough.
  - **Esperas a:** Cristhian (T36, ficha de descriptores).
- [ ] **T52 🟢** ORB y homografía en MATLAB (Computer Vision Toolbox).
  - **Esperas a:** T05 (tuya) y Cristhian (T36).
- [ ] **T53 🟢** `matlab/clasificador_clasico.m` con los mismos umbrales, leídos de `results/clasico/`.
  - **Esperas a:** Cristhian (T41, umbrales).
- [ ] **T54 🟢** Comparar Python frente a MATLAB en 20 imágenes.
  - **Entregable:** tabla de diferencias en `results/matlab/`.
  - **Cómo verificar:** diferencias por debajo de una tolerancia definida y explicada.

### Fase 7. Aprendizaje profundo ⏳ (cuando se vea el tema en clase)

- [ ] **T60 ⏳⛔** Cargar datos desde `data/splits/split.csv` (224×224, normalización).
  - **Esperas a:** Cristhian (T14, división) y el tema de CNN.
- [ ] **T61 ⏳⛔** Modelo: MobileNetV2 congelado y cabeza de 4 clases (o la arquitectura vista en clase).
- [ ] **T62 ⏳⛔** Entrenar con early stopping; guardar curvas de loss y accuracy (train y val) en `results/cnn/`.
- [ ] **T66 ⏳⛔** Comparación pequeña de hiperparámetros (orden: T62 → T66 → T63 → T64).
  - **Qué hacer:** probar tasa de aprendizaje 1e-2, 1e-3 y 1e-4; tamaño de lote 16 y 32; número de capas descongeladas. Comparar las curvas de validación.
  - **Entregable:** `results/cnn/hiperparametros.csv` y sección en `docs/cnn.md` con el porqué de cada valor.
  - **Cómo verificar:** cada valor elegido tiene su curva y su justificación (la rúbrica califica como básico elegirlos al azar).
- [ ] **T63 ⏳⛔** Diagnosticar sobreajuste y ajustar con aumentación de datos y dropout (después de T66).
  - **Cómo verificar:** las curvas de train y val no se separan de forma severa.
- [ ] **T64 ⏳⛔** Evaluar en test con `src/metrics.py` y medir tiempo por frame.
  - **Esperas a:** Cristhian (T25, métricas).
  - **Quién te espera:** Cristhian (T71, auditoría).
- [ ] **T65 ⏳🟢** `docs/cnn.md`: arquitectura, parámetros y decisiones, para explicarlos en la defensa.

### Fase 8. Captura regional y demo

- [ ] **T80 ⛔** Protocolo de captura regional en `docs/captura.md`.
  - **Qué hacer:** cámara fija (webcam o celular) en una escena real de Armenia, Quindío (entrada de una casa, edificio o local, con permiso); cómo simular cada sabotaje (tapar con mano, cinta o tela; desenfocar; girar el soporte); conjunto de iluminación física (luz apagada, lámpara, contraluz de ventana) para escenas normales y saboteadas.
  - **Protección de datos:** cumplir la Ley 1581 de 2012: evitar terceros identificables o pedir su consentimiento, y nunca publicar imágenes con personas.
  - **Cómo verificar:** el protocolo cubre las 4 clases, el conjunto de iluminación y las reglas de protección de datos.
- [ ] **T81 ⛔** Capturar ~100 imágenes por clase y el conjunto de iluminación física en `data/propios/` (no se suben al repositorio).
  - **Cómo verificar:** conteo de ~100 por clase, conjunto de iluminación completo y sin personas identificables.
  - **Quién te espera:** Cristhian (T75, auditoría física); tú mismo (T83); ambos (T93).
- [ ] **T82 🟢** `src/demo.py`: webcam → clasificación clásica en vivo con FPS en pantalla.
  - **Qué hacer:** empezar con un paso de calibración que capture 8 frames normales de referencia de la escena en vivo (el método clásico compara ORB contra referencias de la **misma** cámara; las de Bremen no sirven para otra cámara; 8 como en el artículo).
  - **Esperas a:** Cristhian (T40, reglas); tu tarea T07.
- [ ] **T83 ⛔** Evaluar ambos modelos sobre los datos propios; ajuste fino de la CNN si falla.
  - **Esperas a:** Cristhian (T43); tus tareas T64 y T81.
  - **Quién te espera:** ambos (T93, guía de defensa).
- [ ] **T84 🟢** Agregar la CNN a la demo.
- [ ] **T85 ⛔** Teclas en la demo para activar y desactivar los filtros de T91 (CLAHE, gamma, mediana). Hazla primero con el módulo clásico (depende de T82 y T91); la CNN se agrega cuando T84 esté lista, así el ensayo T94 no espera el tema de CNN.
  - **Qué hacer:** mostrar la predicción antes y después del filtro, con los FPS.
  - **Esperas a:** tus tareas T82 y T91 (T84 solo para agregar la CNN después).
  - **Quién te espera:** ambos (T94, ensayo de la defensa).

### Fase 10. Defensa (tu parte)

- [ ] **T90 ⛔** Diagnóstico de los fallos de la CNN: qué canal o patrón se altera con cada perturbación y por qué.
  - **Esperas a:** Cristhian (T74 y T75, auditoría artificial y física).
- [ ] **T91 ⛔** `src/corrections.py`: CLAHE o ecualización, corrección gamma, mediana, Gaussiano o bilateral.
  - **Cómo verificar:** pruebas de forma y `dtype`.
  - **Quién te espera:** Cristhian (T92, antes/después).

## Avisos de bloqueo

| Cuando termines… | Avisa a Cristhian porque desbloquea… |
|---|---|
| T04 (respuestas de la profesora) | Alcance final de la Fase 6 y confirmación de D2, D3 y D8 |
| T05 (toolboxes) | Decisión de usar ORB u Hough en MATLAB (T52) |
| T20 (perturbaciones) | T70 (conjuntos degradados) |
| T64 (CNN evaluada) | T71 (auditoría de ambos modelos) |
| T81 (captura regional e iluminación física) | T75 (auditoría física) |
| T83 (generalización) | T93 (guía de defensa) |
| T90, parte CNN (diagnóstico) | T93 (guía de defensa) |
| T91 (correcciones) | T92 (antes/después) |
| T85 (filtros en la demo) | T94 (ensayo de la defensa) |

| No empieces… | Hasta que Cristhian termine… |
|---|---|
| Cualquier Pull Request | T02 (colaborador) |
| T12 (galería) | T10 y T11 (descarga e inventario) |
| T44 (análisis de errores) | T43 (evaluación del clásico) |
| T51, T52 (MATLAB) | T36 (ficha de descriptores, incluido el color) |
| T53 (clasificador MATLAB) | T41 (umbrales) |
| T60 (datos CNN) | T14 (división) |
| T64 (evaluación CNN) | T25 (métricas) |
| T82 (demo) | T40 (reglas) |
| T83 (generalización) | T43 (evaluación del clásico) |
| T90 (diagnóstico CNN) | T74 y T75 (auditoría artificial y física) |

## Tareas compartidas

- [ ] **T01 ⛔** Acordar el plan (ver Fase 0).
- [ ] **T03 🟢** Entorno Python (ver Fase 0).
- [ ] **T93 ⛔** `docs/defensa.md`: ataque → síntoma → causa matemática → filtro → mejora medida. Tú escribes la parte de la CNN y de las correcciones.
- [ ] **T94 🟢** Ensayo cruzado con la demo: uno ataca en vivo y el otro explica y activa el filtro, y viceversa (requiere tu T85).
- [ ] **T101 🟢** Tus secciones del informe: perturbaciones, CNN, captura regional, demo y MATLAB.
- [ ] **T102 🟢** Presentación.
- [ ] **T103 ⛔** Revisión cruzada final: revisas las secciones de Cristhian.
