# Plan personal: Santiago Orozco Zuluaga

Derivado del [plan general](PLAN.md). Los IDs, dependencias y tipos son los mismos; si algo cambia, se cambia
primero en el plan general.

**Fecha de entrega:** ≈ 15 de noviembre de 2026 (aproximada, por confirmar). Prioridad de cada tarea: **P1**
imprescindible, **P2** recomendada, **P3** recortable primero (sección 5 del plan general).

## Tu rol

Eres dueño de las perturbaciones, el modelo de aprendizaje profundo, el dataset regional de Medellín (prueba
regional cruzada, D12), la captura en Armenia y la demo en vivo, además de la galería de datos (T12) y el análisis de errores del clásico (T44). Las perturbaciones
(T20) y las correcciones (T91) son la base de la auditoría y de la defensa; la CNN y la demo son lo que se verá en
vivo frente a la profesora. La réplica en MATLAB (Fase 6) quedó como **opcional**, solo si sobra tiempo (D8).

Símbolos: ⛔ bloqueante · 🟢 no bloqueante · ⏳ espera un tema de clase. Cada tarea termina con verificación,
commit y Pull Request revisado por Cristhian. Para subir cambios necesitas que Cristhian termine T02.

## Puedes empezar ya

- T01 Leer y acordar el plan (con Cristhian).
- T03 Configurar el entorno Python.
- T16 Descargar el dataset de Medellín (después de T03).
- T20 Perturbaciones (solo depende de T03; no necesita los datos).
- T07 Entorno local de la demo (después de T03).
- Estudio propio de Keras y transfer learning (D14): no esperes la clase de CNN.
- T80 Protocolo de captura regional.

## Tu calendario

Día 1 = martes 3 de noviembre. Detalle y verificación de capacidad en la sección 6 del plan general
(≈ 59 h de tareas P1 con el estudio de Keras, ≈ 5,5 h por día).

- **Día 1 (mar 3 nov):** T01, T03 (con Cristhian), T07, T20, T21; empezar el estudio de Keras y transfer learning.
- **Día 2 (mié 4 nov):** T80, T81 (captura en Armenia, con el conjunto de luz real), T60 después de que Cristhian termine T14 (si no está, pasa al día 3).
- **Día 3 (jue 5 nov):** T61, T62.
- **Día 4 (vie 6 nov):** T66, T63, T17b (P2; T16 ya está en el PR #2).
- **Día 5 (sáb 7 nov):** T64, T65, T82.
- **Día 6 (dom 8 nov):** T91 primero (T85 la necesita), T85, T84.
- **Día 7 (lun 9 nov):** T16, T17, T18 (P2).
- **Día 8 (mar 10 nov):** T76, T77 (después de T38 de Cristhian), T83 (P2), T67 (P2).
- **Día 9 (mié 11 nov):** ajustes de T91; T90 y T93 con Cristhian.
- **Día 10 (jue 12 nov):** T94 y T101 con Cristhian.
- **Día 11 (vie 13 nov):** T101, T102.
- **Día 12 (sáb 14 nov):** T103 y margen para atrasos.
- **Solo si hay margen:** T12, T22, T23, T44 y MATLAB (T05, T50–T54).

## Lista de tareas

### Fase 0. Organización

- [ ] **T01 ⛔ P1** Leer y acordar el plan con Cristhian.
  - **Qué hacer:** leer el plan general y los dos planes personales; anotar dudas en el PR del plan.
  - **Cómo verificar:** ambos aprueban el PR.
- [ ] **T03 🟢 P1** Configurar el entorno Python (local o Colab) con `requirements.txt`.
  - **Cómo verificar:** `pytest` se ejecuta y `import cv2` funciona.
- [x] **T04 ⛔** Confirmar con la profesora (hecha, 2026-10-29).
  - **Respuesta:** el proyecto puede hacerse solo en Python, solo en MATLAB o en ambos; se acepta Bremen si el informe dice que no se encontró un dataset público regional de sabotaje, que Bremen se usó para entender y desarrollar el trabajo y que el método funcionaría igual con un dataset regional. Decisión del equipo: solo Python, MATLAB opcional (D8).
  - **Pendiente:** registrar la fecha de entrega y los temas pendientes cuando se confirmen.
- [ ] **T05 🟢 Opcional P3** Confirmar versión de MATLAB y toolboxes (Image Processing, Computer Vision, Deep Learning); solo si se hace la Fase 6.
  - **Qué hacer:** ejecutar `ver` en MATLAB y anotar el resultado.
  - **Entregable:** `docs/matlab.md`.
  - **Cómo verificar:** lista de toolboxes escrita.
  - **Quién te espera:** tú mismo (T52, opcional). Si no hay Computer Vision Toolbox, usar Hough en lugar de ORB en MATLAB.
- [ ] **T07 🟢 P1** Entorno local de la demo en el portátil que se llevará a la defensa (D11).
  - **Qué hacer:** instalar Python, OpenCV y TensorFlow (CPU) en el portátil; la demo no corre en Colab porque Colab no accede a la webcam en tiempo real.
  - **Entregable:** instrucciones en `docs/demo.md`.
  - **Cómo verificar:** un script muestra la webcam con los FPS en pantalla.

### Fase 1. Datos (tu parte)

- [ ] **T12 🟢 P3** Galería: 5 imágenes por clase y cámara.
  - **Esperas a:** Cristhian (T10 descarga y T11 inventario).
  - **Qué hacer:** leer `results/inventario.csv` y armar una figura con 5 imágenes por clase y cámara.
  - **Entregable:** `results/figuras/galeria.png`.
  - **Cómo verificar:** la figura muestra todas las clases y cámaras.
- [ ] **T16 ⛔ P2** Descargar el dataset de Medellín.
  - **Qué hacer:** descargar `PedestrianDataSet.zip` (643 MB) desde https://zenodo.org/records/7935299 (DOI 10.5281/zenodo.7935299) a `data/raw/medellin/`; registrar la licencia CC BY 4.0 y la cita (incluido el artículo de datos en Data in Brief, DOI 10.1016/j.dib.2023.109610) en `data/README.md`; inventario: conteo, resolución y cámaras si se pueden identificar por nombre de archivo.
  - **Entregable:** `data/README.md` actualizado, `results/inventario_medellin.csv`.
  - **Cómo verificar:** 6324 imágenes JPG de 1280×720; licencia y cita escritas.
- [ ] **T17 ⛔ P2** Seleccionar ≈600 imágenes normales de Medellín repartidas entre cámaras.
  - **Qué hacer:** excluir, si es posible, las imágenes con personas identificables en primer plano; documentar el criterio en `docs/datos.md`.
  - **Riesgo:** ORB necesita referencias de la **misma** cámara. La cámara no se deduce del nombre de archivo: se recupera con clustering en T17b; como alternativa, evaluar la rotación en Medellín con la orientación de Hough, que no usa referencias.
  - **Cómo verificar:** ≈600 imágenes repartidas entre cámaras, con el criterio escrito.
- [ ] **T17b ⛔ P2** Agrupar las imágenes de Medellín por cámara con clustering no supervisado (clase 20, D15).
  - **Qué hacer:** descriptor global compacto (miniatura en gris de 32×18 o histograma de color); clustering jerárquico de Ward con dendrograma (o K-means con método del codo) para elegir el número de grupos (cerca de 80 cámaras); verificación visual de una muestra de grupos; descarte de grupos con menos de 10 imágenes; apartar 8 referencias por grupo como en `calibration_split`. Resuelve el bloqueo de la revisión del PR #2: Medellín no tiene referencias de la misma cámara y el módulo clásico las necesita.
  - **Entregable:** `src/medellin.py` (o un módulo nuevo), `data/splits/medellin_grupos.csv`, nota en `docs/datos.md`.
  - **Cómo verificar:** dendrograma (o curva del codo) guardado con el número de grupos justificado; muestra de grupos revisada a mano; cada grupo conservado tiene 10 o más imágenes y 8 referencias apartadas.
  - **Esperas a:** tu tarea T16 (en el PR #2).
  - **Quién te espera:** tú mismo (T18, T76).
- [ ] **T18 ⛔ P2** `src/synthetic_tampering.py`: sabotaje sintético sobre el subconjunto.
  - **Qué hacer:** generar desenfocada (desenfoque gaussiano), girada (rotación con recorte) y tapada (oclusión total y parcial) con semilla fija, como en el dataset UHCTD; salida en `data/medellin_sintetico/` (no versionado) con un script reproducible.
  - **Entregable:** `src/synthetic_tampering.py`, `tests/test_synthetic_tampering.py`.
  - **Cómo verificar:** `pytest` pasa y la misma semilla regenera el mismo conjunto.
  - **Esperas a:** tus tareas T17 y T17b.
  - **Quién te espera:** tú mismo (T76).

### Fase 2. Perturbaciones

- [ ] **T20 ⛔ P1** `src/perturbations.py`.
  - **Qué hacer:** funciones de brillo (oscurecer y aclarar), corrección gamma, ruido gaussiano, sal y pimienta y compresión JPEG; 3 niveles cada una; parámetro de semilla. **No incluir desenfoque** (es una clase, ver D7).
  - **Entregable:** `src/perturbations.py`.
  - **Cómo verificar:** las pruebas de T21 pasan.
  - **Quién te espera:** Cristhian (T70, conjuntos degradados); tú mismo (T91; T50 si se hace MATLAB).
- [ ] **T21 ⛔ P1** `tests/test_perturbations.py`.
  - **Qué hacer:** probar que cada perturbación conserva forma, `dtype` uint8, rango 0–255 y es reproducible con la misma semilla.
  - **Cómo verificar:** `pytest` pasa.
- [ ] **T22 🟢 P2** `notebooks/02_perturbaciones.ipynb`: grilla perturbación × nivel.
- [ ] **T23 🟢 P2** Justificar los niveles: qué situación real simula cada uno (anochecer, contraluz, sensor ruidoso, transmisión comprimida…).

### Fase 5. Clásico (tu parte)

- [ ] **T44 🟢 P2** `notebooks/04_clasico.ipynb`: análisis de errores del módulo clásico.
  - **Esperas a:** Cristhian (T43, evaluación del clásico).
  - **Qué hacer:** mostrar las imágenes mal clasificadas, agruparlas por tipo de error y escribir la causa (qué descriptor falló).
  - **Cómo verificar:** cada tipo de error tiene ejemplos y causa escrita.

### Fase 6. MATLAB (opcional, solo si sobra tiempo)

No cuenta en la ruta crítica ni en los totales de horas (≈ 12 h aparte). Si se hace, la ficha T36 de Cristhian ya
estará escrita para el informe; no hace falta esperarla como bloqueo.


- [ ] **T50 🟢 Opcional P3** `matlab/perturbaciones.m`, equivalente a T20 (mismos parámetros y niveles).
- [ ] **T51 🟢 Opcional P3** `matlab/descriptores.m`: Laplaciano, desviación, entropía, color HSV, Canny y Hough.
  - **Usa:** la ficha `docs/descriptores.md` (T36).
- [ ] **T52 🟢 Opcional P3** ORB y homografía en MATLAB (Computer Vision Toolbox).
  - **Usa:** T05 (tuya) y la ficha de T36.
- [ ] **T53 🟢 Opcional P3** `matlab/clasificador_clasico.m` con los mismos umbrales, leídos de `results/clasico/`.
  - **Usa:** los umbrales de `results/clasico/umbrales.json` (T41).
- [ ] **T54 🟢 Opcional P3** Comparar Python frente a MATLAB en 20 imágenes.
  - **Entregable:** tabla de diferencias en `results/matlab/`.
  - **Cómo verificar:** diferencias por debajo de una tolerancia definida y explicada.

### Fase 7. Aprendizaje profundo (⏳ ajustable: empieza ya y ajusta con la clase, D14)

- [ ] **T60 ⏳ ajustable ⛔ P1** Cargar datos desde `data/splits/split.csv` (224×224, normalización).
  - **Esperas a:** Cristhian (T14, división). No esperes el tema de CNN: estudia Keras por tu cuenta y ajusta cuando se vea en clase.
- [ ] **T61 ⏳ ajustable ⛔ P1** Modelo: MobileNetV2 congelado y cabeza de 4 clases (o la arquitectura vista en clase).
- [ ] **T62 ⏳ ajustable ⛔ P1** Entrenar con early stopping; guardar curvas de loss y accuracy (train y val) en `results/cnn/`.
- [ ] **T66 ⏳ ajustable ⛔ P1** Comparación pequeña de hiperparámetros (orden: T62 → T66 → T63 → T64).
  - **Qué hacer:** probar tasa de aprendizaje 1e-2, 1e-3 y 1e-4; tamaño de lote 16 y 32; número de capas descongeladas. Comparar las curvas de validación.
  - **Entregable:** `results/cnn/hiperparametros.csv` y sección en `docs/cnn.md` con el porqué de cada valor.
  - **Cómo verificar:** cada valor elegido tiene su curva y su justificación (la rúbrica califica como básico elegirlos al azar).
- [ ] **T63 ⏳ ajustable ⛔ P1** Diagnosticar sobreajuste y ajustar con aumentación de datos y dropout (después de T66).
  - **Cómo verificar:** las curvas de train y val no se separan de forma severa.
- [ ] **T64 ⏳ ajustable ⛔ P1** Evaluar en test y en la cámara no vista (ubicación 2) con `src/metrics.py` y medir tiempo por frame.
  - **Esperas a:** Cristhian (T25, métricas; T19, cámara no vista).
  - **Quién te espera:** Cristhian (T71, auditoría).
- [ ] **T65 ⏳ ajustable 🟢 P1** `docs/cnn.md`: arquitectura, parámetros y decisiones, para explicarlos en la defensa.
- [ ] **T67 🟢 P2** Explicabilidad de la CNN con Grad-CAM (o LIME/SHAP).
  - **Qué hacer:** mapas de calor sobre al menos 4 ejemplos por clase en test y en la cámara no vista; escribir si el modelo mira el sabotaje (la tapa, el desenfoque, el cambio de escena) o el contexto (iluminación, cielo, vegetación). Es lo que pide la actividad evaluativa 3 de IA (SHAP y LIME sobre el contexto).
  - **Entregable:** `notebooks/05_explicabilidad_cnn.ipynb`, `results/figuras/gradcam_*.png` y una sección en `docs/cnn.md`.
  - **Cómo verificar:** cada figura con su conclusión; la sección explica la caída de F1 de 0,97 (test) a 0,58 (cámara no vista).
  - **Esperas a:** tu T64. **Quién te espera:** T90 y T93 (diagnóstico y guía de la defensa, ambos).

### Fase 8. Captura en Armenia y demo

- [ ] **T80 ⛔ P1** Protocolo de captura regional en `docs/captura.md`.
  - **Qué hacer:** cámara fija (webcam o celular) en una escena real de Armenia, Quindío (entrada de una casa, edificio o local, con permiso); cómo simular cada sabotaje (tapar con mano, cinta o tela; desenfocar; girar el soporte); conjunto de iluminación física (luz apagada, lámpara, contraluz de ventana) para escenas normales y saboteadas.
  - **Protección de datos:** cumplir la Ley 1581 de 2012: evitar terceros identificables o pedir su consentimiento, y nunca publicar imágenes con personas.
  - **Cómo verificar:** el protocolo cubre las 4 clases, el conjunto de iluminación y las reglas de protección de datos.
- [ ] **T81 ⛔ P1** Capturar ~100 imágenes por clase y el conjunto de iluminación física en `data/propios/` (no se suben al repositorio).
  - **Cómo verificar:** conteo de ~100 por clase, conjunto de iluminación completo y sin personas identificables.
  - **Quién te espera:** Cristhian (T75, auditoría física); tú mismo (T83); ambos (T93).
- [ ] **T82 🟢 P1** `src/demo.py`: webcam → clasificación clásica en vivo con FPS en pantalla.
  - **Qué hacer:** empezar con un paso de calibración que capture 8 frames normales de referencia de la escena en vivo (el método clásico compara ORB contra referencias de la **misma** cámara; las de Bremen no sirven para otra cámara; 8 como en el artículo).
  - **Esperas a:** Cristhian (T40, reglas); tu tarea T07.
- [ ] **T83 ⛔ P2** Evaluar ambos modelos sobre los datos propios; ajuste fino de la CNN si falla.
  - **Esperas a:** Cristhian (T43); tus tareas T64 y T81.
  - **Quién te espera:** ambos (T93, guía de defensa).
- [ ] **T84 🟢 P1** Agregar la CNN a la demo.
- [ ] **T85 ⛔ P1** Teclas en la demo para activar y desactivar los filtros de T91 (CLAHE, gamma, mediana). Hazla primero con el módulo clásico (depende de T82 y T91); la CNN se agrega cuando T84 esté lista, así el ensayo T94 no espera el tema de CNN.
  - **Qué hacer:** mostrar la predicción antes y después del filtro, con los FPS.
  - **Esperas a:** tus tareas T82 y T91 (T84 solo para agregar la CNN después).
  - **Quién te espera:** ambos (T94, ensayo de la defensa).

### Fase 9. Auditoría (tu parte)

- [ ] **T76 ⛔ P2** Evaluación regional cruzada: entrenados en Bremen, evaluados en Medellín.
  - **Esperas a:** Cristhian (T43, clásico evaluado); tus tareas T17b, T18 y T64, y las correcciones de la revisión del PR #2.
  - **Qué hacer:** matriz de confusión, F1 y tasa de falsos positivos de ambos modelos en Medellín; comparar con el resultado en Bremen y explicar la diferencia.
  - **Entregable:** `results/regional/` y sección en `notebooks/06_auditoria.ipynb`.
  - **Quién te espera:** Cristhian (T90, diagnóstico del clásico; T101, informe); tú mismo (T90, CNN).
- [ ] **T77 🟢 P2** Robustez de ORB, SIFT y KAZE ante las perturbaciones.
  - **Esperas a:** Cristhian (T38, comparación de descriptores); tu tarea T20.
  - **Qué hacer:** buenas coincidencias frente a severidad para brillo, gamma, ruido y JPEG. Comprueba lo que dice la clase 11 (ORB pierde precisión con luz y ruido; KAZE es más robusto).
  - **Entregable:** `results/auditoria/descriptores_vs_perturbacion.csv` y figura.
  - **Quién te espera:** ambos (T90, T93): argumento de defensa "con poca luz ORB pierde coincidencias → aplicar CLAHE o cambiar a KAZE".

### Fase 10. Defensa (tu parte)

- [ ] **T90 ⛔ P1** Diagnóstico de los fallos de la CNN: qué canal o patrón se altera con cada perturbación y por qué.
  - **Esperas a:** Cristhian (T74 y T75, auditoría artificial y física).
- [ ] **T91 ⛔ P1** `src/corrections.py`: CLAHE o ecualización, corrección gamma, mediana, Gaussiano o bilateral.
  - **Cómo verificar:** pruebas de forma y `dtype`.
  - **Quién te espera:** Cristhian (T92, antes/después).

## Avisos de bloqueo

| Cuando termines… | Avisa a Cristhian porque desbloquea… |
|---|---|
| T20 (perturbaciones) | T70 (conjuntos degradados) |
| T64 (CNN evaluada) | T71 (auditoría de ambos modelos) |
| T81 (captura regional e iluminación física) | T75 (auditoría física) |
| T83 (generalización) | T93 (guía de defensa) |
| T90, parte CNN (diagnóstico) | T93 (guía de defensa) |
| T91 (correcciones) | T92 (antes/después) |
| T85 (filtros en la demo) | T94 (ensayo de la defensa) |
| T76 (prueba regional en Medellín; antes necesitas tu T17b y las correcciones de la revisión del PR #2) | T90 (diagnóstico del clásico) y T101 (informe) |
| T77 (robustez de descriptores) | T90 y T93 (argumento de defensa sobre descriptores) |

| No empieces… | Hasta que Cristhian termine… |
|---|---|
| Cualquier Pull Request | T02 (colaborador) |
| T12 (galería) | T10 y T11 (descarga e inventario) |
| T44 (análisis de errores) | T43 (evaluación del clásico) |
| T60 (datos CNN) | T14 (división) |
| T64 (evaluación CNN) | T25 (métricas) y T19 (cámara no vista) |
| T77 (robustez de descriptores) | T38 (comparación de descriptores) |
| T82 (demo) | T40 (reglas) |
| T76 (prueba regional) y T83 (generalización) | T43 (evaluación del clásico) |
| T90 (diagnóstico CNN) | T74 y T75 (auditoría artificial y física) |

## Tareas compartidas

- [ ] **T01 ⛔ P1** Acordar el plan (ver Fase 0).
- [ ] **T03 🟢 P1** Entorno Python (ver Fase 0).
- [ ] **T93 ⛔ P1** `docs/defensa.md`: ataque → síntoma → causa matemática → filtro → mejora medida. Tú escribes la parte de la CNN y de las correcciones.
- [ ] **T94 🟢 P1** Ensayo cruzado con la demo: uno ataca en vivo y el otro explica y activa el filtro, y viceversa (requiere tu T85).
- [ ] **T101 🟢 P1** Tus secciones del informe: perturbaciones, CNN, dataset regional de Medellín, captura en Armenia, demo y MATLAB si se hizo.
- [ ] **T102 🟢 P1** Presentación.
- [ ] **T103 ⛔ P1** Revisión cruzada final: revisas las secciones de Cristhian.
