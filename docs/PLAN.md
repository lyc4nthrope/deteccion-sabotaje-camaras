# Plan del proyecto final: detección de sabotaje en cámaras de vigilancia

**Grupo 7 · Seguridad** · Cristhian Eduardo Osorio Restrepo y Santiago Orozco Zuluaga
**Docente:** Ing. Ana Maria Tamayo Ocampo, Mg · **Fecha de entrega:** _por definir_

> Este plan es un documento vivo. Las fases marcadas con ⏳ dependen de temas que aún no se han visto en
> clase; se ajustan cuando la profesora los explique (ver sección 4).

---

## 1. Requisitos frente al estado actual

| # | Requisito (guía del proyecto) | Estado actual | Evidencia |
|---|---|---|---|
| R1 | Problema regional de clasificación o detección, temática distinta por grupo | ✅ Definido: clasificar imágenes de cámaras de vigilancia saboteadas | Acuerdo del grupo (2026-10-29) |
| R2 | Proceso en Python **y** MATLAB | ⬜ Pendiente | — |
| R3 | Módulo 1, clásico: descriptores, transformaciones geométricas, umbralización, filtros lineales y no lineales. Métricas: tiempo por frame y tasa de falsos positivos en entorno estable | ⬜ Pendiente. Ya vistos en clase: filtros, bordes, morfología, segmentación, texturas, momentos de Hu, Hough | Notebooks de clase en `context_vision/` |
| R4 | Módulo 2, aprendizaje profundo: CNN propia o transfer learning. Métricas: curvas de loss y accuracy | ⏳ Depende del tema de CNN en clase | — |
| R5 | Módulo 3, auditoría de robustez: ambos sistemas ante entornos degradados. Métricas: matrices de confusión cruzadas, ROC y F1 en condiciones críticas | ⬜ Pendiente | — |
| R6 | Defensa ante ataque adversario en vivo (40 % de la nota): diagnosticar el fallo y proponer el filtro correctivo | ⬜ Pendiente | — |
| R7 | Dataset | ✅ Encontrado y accesible: ComNets Camera Tampering Images (Universidad de Bremen), 3600 imágenes: 1800 normales, 600 desenfocadas, 600 giradas, 600 con lente tapado | Enlace verificado (HTTP 200): https://nc.uni-bremen.de/index.php/s/GCsBcswrQf9kBXC · Artículo: Attarha, Shanmugi y Förster, *Camera Tampering Detection for Outdoor IoT Systems*, EWSN 2024 (arXiv 2602.05706) |

**Brecha principal:** existe una referencia (el artículo de Bremen compara reglas frente a CNN), pero no prueba
los modelos ante cambios de luz ni ruido. Ese es el aporte de este proyecto: la auditoría de robustez y la defensa.

---

## 2. Decisiones

| ID | Decisión | Elección recomendada | Por qué |
|---|---|---|---|
| D1 | Clases | 4 clases: normal, desenfocada, girada, tapada. Además se reporta la versión binaria (normal frente a saboteada) | Las 4 clases permiten diagnosticar el tipo de sabotaje; la versión binaria da una curva ROC directa |
| D2 | Dataset principal | ComNets Bremen, citando el artículo. Confirmar la licencia al descargar | Imágenes reales, etiquetadas y de tamaño manejable |
| D3 | Datos propios | Captura pequeña con webcam o celular fijo (~100 imágenes por clase) | El artículo advierte que la CNN falla si cambia la cámara. La demo en vivo usará otra cámara, así que se necesita prueba y ajuste con datos propios. Aporta el componente regional |
| D4 | Módulo clásico | Reglas con umbrales calibrados sobre los datos de entrenamiento, usando descriptores de clase: varianza del Laplaciano (nitidez), desviación estándar e histograma (tapado), densidad de bordes Canny, coincidencias ORB con imágenes de referencia y ángulo por homografía o rectas de Hough (giro) | Sigue el método validado del artículo y usa los temas de la materia. La rúbrica pide justificar los umbrales |
| D5 | Módulo profundo | ⏳ Transfer learning con un modelo liviano (por ejemplo MobileNetV2 en Keras), como opción por defecto. Se ajusta a lo que se vea en clase | Pocos datos y poco tiempo: el transfer learning converge rápido y reduce el sobreajuste |
| D6 | División de datos | Por bloques de tiempo o ubicación de cámara, no aleatoria imagen por imagen | Las imágenes consecutivas son casi idénticas: una división aleatoria infla las métricas (fuga de datos) |
| D7 | Perturbaciones de la auditoría | Brillo (oscurecer y aclarar), ruido gaussiano, sal y pimienta, desenfoque leve y compresión JPEG, cada una en 3 niveles, con semilla fija | Son los ataques probables en la defensa (luz y ruido). Los niveles permiten curvas de F1 frente a severidad |
| D8 | Reparto Python / MATLAB | Python: todo el flujo. MATLAB: módulo clásico y perturbaciones (la CNN en MATLAB es opcional) | Cumple R2 sin duplicar el entrenamiento de la red |
| D9 | Métricas | Tiempo por frame, tasa de falsos positivos, curvas de loss y accuracy, matrices de confusión cruzadas, ROC (binaria y uno contra todos) y F1 macro | Son exactamente las exigidas por la guía |

---

## 3. Fases

Cada fase cabe en una sesión, termina verificada (pruebas o ejecución real), con su commit y su registro en
Engram, y se detiene en un punto de control antes de la siguiente.

| Fase | Alcance | Decisiones | Verificación | Estado |
|---|---|---|---|---|
| **F0** | Repositorio, estructura y plan | — | El repositorio existe y el plan está revisado por el grupo | ✅ |
| **F1** | Descarga y exploración del dataset: licencia, estructura, conteos, resolución, muestra visual y archivo de división | D2, D6 | Los conteos coinciden con 1800/600/600/600; el archivo de división se guarda y no hay imágenes repetidas entre conjuntos | ⬜ |
| **F2** | Módulo de perturbaciones en Python (funciones reutilizables + figura de ejemplo) | D7 | Pruebas: cada perturbación conserva forma y tipo de la imagen y es reproducible con la misma semilla | ⬜ |
| **F3** | Módulo clásico, parte 1: cálculo de descriptores y su análisis por clase | D4 | Gráficas de caja por descriptor que muestran qué descriptor separa cada clase | ⬜ |
| **F4** | Módulo clásico, parte 2: reglas, calibración de umbrales y evaluación | D4, D9 | Matriz de confusión, tasa de falsos positivos y tiempo por frame sobre el conjunto de prueba | ⬜ |
| **F5** | Réplica en MATLAB del módulo clásico y las perturbaciones | D8 | Mismos descriptores en 20 imágenes, con diferencias menores a una tolerancia definida | ⬜ |
| **F6** ⏳ | Módulo profundo: entrenamiento, curvas de loss y accuracy, evaluación | D5, D6, D9 | Curvas sin sobreajuste severo y métricas sobre el conjunto de prueba | ⬜ |
| **F7** | Captura propia y demo en vivo con webcam (ambos modelos, mostrando FPS) | D3 | La demo clasifica en vivo y se mide el acierto sobre la captura propia | ⬜ |
| **F8** | Auditoría de robustez: ambos modelos sobre los conjuntos degradados | D7, D9 | Matrices de confusión cruzadas, ROC, F1 frente a severidad y tiempos | ⬜ |
| **F9** | Preparación de la defensa: diagnóstico de cada fallo y filtros correctivos (por ejemplo CLAHE, corrección gamma, mediana) re-evaluados | D7 | Por cada perturbación: explicación del fallo, filtro propuesto y mejora medida | ⬜ |
| **F10** | Informe técnico y presentación | — | Revisión completa de los dos integrantes | ⬜ |

F1 a F5 se pueden hacer ya con lo visto en clase. F7 puede adelantarse en paralelo (es captura de datos).

---

## 4. Temas de la materia que faltan

Cuando se vea un tema nuevo, se revisa esta tabla y se ajustan las fases y decisiones afectadas.

| Tema | Estado en clase | Fases afectadas | Qué podría cambiar |
|---|---|---|---|
| Filtros, bordes, morfología, segmentación, texturas, momentos de Hu, Hough | ✅ Visto | F3, F4, F5 | — |
| Redes neuronales convolucionales y transfer learning | ⏳ Por ver | F6, F8 | D5: arquitectura y framework |
| Métricas de evaluación (ROC, F1) | ⏳ Confirmar si se ve en clase | F4, F8 | D9: forma de presentarlas |
| Otros temas que anuncie la profesora | ⏳ | Por definir | Por definir |

---

## 5. Riesgos

| Riesgo | Mitigación |
|---|---|
| La licencia del dataset no está escrita en la página de descarga | Citar el artículo, usarlo solo con fines académicos y confirmar en F1 |
| La CNN no generaliza a la cámara de la demo (lo advierte el propio artículo) | Captura propia (D3) para ajuste fino y prueba |
| Fuga de datos por imágenes casi idénticas | División por bloques de tiempo o ubicación (D6) |
| Poco tiempo | Fases pequeñas; F1 a F5 no dependen de temas pendientes |
