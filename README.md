# Detección de sabotaje en cámaras de vigilancia

Proyecto final de **Visión Artificial** · Grupo 7 · Temática: **Seguridad**

**Integrantes:** Cristhian Eduardo Osorio Restrepo y Santiago Orozco Zuluaga
**Docente:** Ing. Ana Maria Tamayo Ocampo, Mg

## Problema

Antes de un robo o un acto vandálico, es común que las cámaras de vigilancia sean **tapadas, desenfocadas o
giradas** para dejar de registrar la escena. Si nadie lo detecta a tiempo, el sistema de seguridad queda ciego
aunque siga encendido. Detectar este sabotaje de forma automática permite alertar al personal de seguridad en
cuanto ocurre.

## Contexto regional

Las cámaras de vigilancia son una herramienta clave de seguridad en las ciudades colombianas, y en Armenia
(Quindío) apoyan el monitoreo y la captura de delincuentes; sin embargo, muchas quedan fuera de servicio o son
vandalizadas. No existe un dataset público regional de cámaras saboteadas, por lo que el proyecto entrena con un
dataset internacional y valida con una captura propia en una escena real de Armenia. Las fuentes verificadas
están en `docs/contexto_regional.md` (tarea T06).

## Qué vamos a hacer

Clasificar cada imagen de una cámara en una de **4 clases: normal, desenfocada, girada o tapada**, y reportar
además la versión binaria (normal frente a saboteada).

- **Módulo 1, clásico (20 %):** clasificador por reglas con descriptores matemáticos y umbrales calibrados y
  justificados. Se mide el tiempo de cómputo por frame y la tasa de falsos positivos en entornos estables.
- **Módulo 2, aprendizaje profundo (20 %):** red neuronal convolucional con transfer learning. Se reportan las
  curvas de loss y accuracy en entrenamiento y validación, sin sobreajuste severo.
- **Módulo 3, auditoría de robustez (20 %):** ambos modelos se evalúan con imágenes degradadas por cambios de luz
  y ruido. Se reportan matrices de confusión cruzadas, curvas ROC y F1-score en condiciones críticas.
- **Defensa (40 %):** la profesora ataca el modelo en vivo con cambios de luz o ruido; debemos diagnosticar
  matemáticamente el fallo y proponer el filtro exacto que lo corrige.

Todo el proceso se implementa en **Python y MATLAB**.

## Cómo lo vamos a hacer

1. **Datos:** se entrena con imágenes reales de cámaras exteriores (ver [Dataset](#dataset)) y se valida con una
   **captura propia regional** en Armenia, Quindío, que también incluye cambios reales de iluminación. La captura
   respeta la Ley 1581 de 2012 de protección de datos personales.
2. **División por bloques:** train/val/test 70/15/15 por bloques de tiempo o ubicación de cámara, no aleatoria,
   porque los frames consecutivos son casi idénticos y una división aleatoria inflaría las métricas.
3. **Módulo clásico:** preprocesamiento (escala de grises, tamaño fijo, filtro Gaussiano y mediana) y descriptores:
   - Varianza del Laplaciano → nitidez (detecta **desenfocada**).
   - Desviación estándar y entropía del histograma → uniformidad (detecta **tapada**).
   - Coincidencias ORB contra imágenes de referencia y ángulo por homografía → geometría (detecta **girada**).
   - Saturación, dispersión del matiz y proporción de tono de piel en HSV → umbralización de color (separa
     **tapada** con la mano de una escena nocturna oscura).
   - Densidad de bordes Canny y orientación dominante de Hough → descriptores complementarios.

   Un árbol de reglas (tapada → desenfocada → girada → normal) decide la clase, y un puntaje continuo de
   sabotaje permite trazar la curva ROC.
4. **Módulo CNN** ⏳: transfer learning con MobileNetV2 y una cabeza de 4 clases (se confirma cuando se vea el
   tema en clase), con una comparación pequeña de hiperparámetros para justificar cada valor.
5. **Auditoría:** perturbaciones de brillo (oscurecer y aclarar), corrección gamma, ruido gaussiano, sal y
   pimienta y compresión JPEG, en 3 niveles y con semilla fija (degradación artificial), más la captura con
   cambios reales de luz: luz apagada, lámpara y contraluz de ventana (degradación física). El desenfoque no se
   usa como perturbación porque es una de las clases.
6. **Correcciones para la defensa:** CLAHE o ecualización, corrección gamma, filtro de mediana, Gaussiano o
   bilateral, con la mejora medida antes y después.
7. **Demo en vivo:** corre localmente en el portátil con la webcam (Colab no accede a la cámara en tiempo real).
   Al iniciar, se calibra con 8 frames normales de la escena como referencias; durante la defensa, unas teclas
   activan los filtros correctivos y muestran la predicción antes y después, con los FPS.

## Línea base de referencia

El artículo del dataset compara un método por reglas con una CNN binaria (normal frente a anormal):

| Método | Accuracy | Tiempo por imagen | Tamaño del modelo |
|---|---|---|---|
| Reglas (ORB, homografía, Laplaciano, desviación estándar) | 0,905 | 0,07 s | 8,77 KB |
| CNN (Conv 32-64-128, densa 128, salida sigmoide) | 0,9975 | 0,02 s | 37,8 MB |

El artículo no evalúa la robustez ante cambios de iluminación ni ruido, y advierte que la CNN necesita datos
nuevos si la cámara cambia de posición. **La auditoría de robustez (artificial y física), la validación con una
captura propia en Armenia y la defensa son el aporte de este proyecto.**

## Tecnologías

La guía del proyecto exige realizar el proceso en **Python y MATLAB**. Python lleva el flujo completo y MATLAB
replica el módulo clásico para comprobar que ambos dan el mismo resultado. La red se entrena en Colab y la demo
en vivo corre localmente en el portátil con TensorFlow en CPU, porque Colab no accede a la webcam en tiempo real.

### Lenguajes y entornos

| Tecnología | Para qué la usamos | Por qué esta |
|---|---|---|
| **Python 3.11** | Todo el flujo: datos, módulo clásico, red neuronal, auditoría y demo en vivo | Es el lenguaje con más herramientas de visión artificial y aprendizaje profundo, y es el que se usa en los notebooks de clase. La versión 3.11 es compatible con todas las librerías de esta lista |
| **MATLAB** (Image Processing Toolbox y Computer Vision Toolbox) | Réplica de las perturbaciones y del módulo clásico (Fase 6) | Lo exige la guía. Las dos toolboxes traen los mismos algoritmos vistos en clase (Laplaciano, Canny, Hough, ORB y homografía), así que la réplica es directa. Si falta el Computer Vision Toolbox, la rotación se detecta con Hough (tarea T05) |
| **Jupyter / Google Colab** | Notebooks de análisis y entrenamiento de la red | Los notebooks dejan el código, las gráficas y las explicaciones juntos, que es el formato de entrega de la materia. Colab da **GPU gratis**, necesaria para entrenar la red en minutos y no en horas |
| **Git y GitHub** | Control de versiones, ramas y Pull Requests | Permite trabajar los dos en paralelo sin pisarse, revisar el trabajo del otro antes de unirlo y volver a una versión anterior si algo se daña |

### Librerías de Python

| Librería | Para qué la usamos | Por qué esta |
|---|---|---|
| **OpenCV** | Lectura de imágenes, filtros (Gaussiano, mediana), Laplaciano, Canny, ORB, homografía, Hough y captura de la webcam | Tiene en una sola librería todos los algoritmos del módulo clásico, es muy rápida (importa para medir el **tiempo por frame**) y maneja la cámara en vivo para la demo. Es la misma que se usa en clase y la que usó el artículo de referencia |
| **NumPy** | Operaciones sobre las matrices de píxeles | Una imagen es una matriz de números, y NumPy es la base de todas las demás librerías. Permite calcular descriptores como la desviación estándar en una línea |
| **scikit-image** | Entropía del histograma y funciones de análisis que OpenCV no trae | Complementa a OpenCV con funciones de análisis bien documentadas, ya usadas en las guías de la materia |
| **TensorFlow / Keras** | Red neuronal: transfer learning con MobileNetV2 y curvas de loss y accuracy | Keras arma una red en pocas líneas, cada una corresponde a una capa, y eso facilita **explicar la arquitectura** en la defensa (la rúbrica castiga usar un modelo como caja negra). Entrega las curvas de loss y accuracy directamente, viene instalada en Colab y fue la usada por el artículo de referencia, así que los resultados se pueden comparar. Se descartó PyTorch porque exige escribir el ciclo de entrenamiento a mano, y el Deep Learning Toolbox de MATLAB porque no tiene GPU gratis |
| **MobileNetV2** (dentro de Keras) | Modelo preentrenado para el transfer learning | Es una red **liviana**: entrena rápido con pocas imágenes (3600) y clasifica en pocos milisegundos, importante para la demo en vivo. Su arquitectura se ajusta a lo que se vea en el tema de CNN en clase |
| **scikit-learn** | Matriz de confusión, curva ROC y AUC, F1 | Calcula con funciones probadas exactamente las métricas que exige la rúbrica, iguales para el módulo clásico y la red, así la comparación es justa |
| **pandas** | Tablas de inventario del dataset y de resultados por condición | La auditoría genera muchas combinaciones (modelo × perturbación × nivel). pandas las organiza en tablas que se filtran, comparan y exportan al informe |
| **Matplotlib** | Gráficas: diagramas de caja, curvas de loss y accuracy, ROC, F1 frente a severidad | Es la librería estándar de gráficas en Python y la usada en los notebooks de clase |
| **pytest** | Pruebas automáticas de cada módulo | Cada tarea se verifica antes de unirla a `main`. Por ejemplo, comprueba que las perturbaciones sean reproducibles o que la división de datos no repita imágenes entre conjuntos |

## Estructura del repositorio

```
data/
  README.md        Origen de los datos y condiciones de uso
  raw/             Dataset descargado (no se versiona)
  propios/         Captura propia (no se versiona)
  splits/          División train/val/test (sí se versiona)
docs/
  PLAN.md          Plan general
  PLAN_CRISTHIAN.md
  PLAN_SANTIAGO.md
  contexto_regional.md  Justificación regional con fuentes (T06)
notebooks/         Notebooks de Python (compatibles con Colab)
src/               Código Python reutilizable
tests/             Pruebas automáticas (pytest)
matlab/            Implementación en MATLAB
results/           Figuras y métricas generadas
informe/           Informe técnico y presentación
```

## Dataset

**ComNets Camera Tampering Images Datasets** (Universidad de Bremen): 3600 imágenes, 1800 normales,
600 desenfocadas, 600 giradas y 600 con lente tapado.

- **Descarga:** https://nc.uni-bremen.de/index.php/s/GCsBcswrQf9kBXC
- **Uso:** la página no indica licencia; se usa solo con fines académicos y citando el artículo.
- **No se sube al repositorio** (ver [data/README.md](data/README.md)).
- **Captura propia regional:** ~100 imágenes por clase en Armenia, Quindío, más un conjunto con cambios reales
  de iluminación (`data/propios/`, no se versiona).

> S. Attarha, K. Shanmugi y A. Förster, "Poster: Camera Tampering Detection for Outdoor IoT Systems",
> EWSN 2024. arXiv:2602.05706.

## Cómo trabajamos

- Una rama por tarea, con el ID del plan: por ejemplo `feat/T14-division-datos`.
- Cada tarea se integra con un Pull Request a `main`, revisado por el otro integrante.
- Commits convencionales (`feat:`, `fix:`, `docs:`, `test:`…).
- `main` siempre debe ejecutarse sin errores.
- Nunca se suben datos ni modelos entrenados.

## Plan

- [Plan general](docs/PLAN.md): requisitos, decisiones, ruta crítica y todas las tareas.
- [Plan de Cristhian](docs/PLAN_CRISTHIAN.md): contexto regional, datos, métricas, módulo clásico y auditoría.
- [Plan de Santiago](docs/PLAN_SANTIAGO.md): perturbaciones, MATLAB, CNN, captura regional y demo.

## Estado

Fase 0 (organización) lista: repositorio, estructura y plan. Siguiente paso: descarga y división de los datos.
