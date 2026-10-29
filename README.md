# Detección de sabotaje en cámaras de vigilancia

Proyecto final de **Visión Artificial** · Grupo 7 · Temática: **Seguridad**

**Integrantes:** Cristhian Eduardo Osorio Restrepo y Santiago Orozco Zuluaga
**Docente:** Ing. Ana Maria Tamayo Ocampo, Mg

## Problema

Antes de un robo o un acto vandálico, es común que las cámaras de vigilancia sean **tapadas, desenfocadas o
giradas** para dejar de registrar la escena. Si nadie lo detecta a tiempo, el sistema de seguridad queda ciego
aunque siga encendido. Detectar este sabotaje de forma automática permite alertar al personal de seguridad en
cuanto ocurre.

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

1. **Dataset:** imágenes reales de cámaras exteriores (ver [Dataset](#dataset)), más una captura propia pequeña
   con webcam o celular para la demo y para medir la generalización a otra cámara.
2. **División por bloques:** train/val/test 70/15/15 por bloques de tiempo o ubicación de cámara, no aleatoria,
   porque los frames consecutivos son casi idénticos y una división aleatoria inflaría las métricas.
3. **Módulo clásico:** preprocesamiento (escala de grises, tamaño fijo, filtro Gaussiano y mediana) y descriptores:
   - Varianza del Laplaciano → nitidez (detecta **desenfocada**).
   - Desviación estándar y entropía del histograma → uniformidad (detecta **tapada**).
   - Coincidencias ORB contra imágenes de referencia y ángulo por homografía → geometría (detecta **girada**).
   - Densidad de bordes Canny y orientación dominante de Hough → descriptores complementarios.

   Un árbol de reglas (tapada → desenfocada → girada → normal) decide la clase, y un puntaje continuo de
   sabotaje permite trazar la curva ROC.
4. **Módulo CNN** ⏳: transfer learning con MobileNetV2 y una cabeza de 4 clases (se confirma cuando se vea el
   tema en clase).
5. **Auditoría:** perturbaciones de brillo (oscurecer y aclarar), corrección gamma, ruido gaussiano, sal y
   pimienta y compresión JPEG, en 3 niveles y con semilla fija. El desenfoque no se usa como perturbación porque
   es una de las clases.
6. **Correcciones para la defensa:** CLAHE o ecualización, corrección gamma, filtro de mediana, Gaussiano o
   bilateral, con la mejora medida antes y después.

## Línea base de referencia

El artículo del dataset compara un método por reglas con una CNN binaria (normal frente a anormal):

| Método | Accuracy | Tiempo por imagen | Tamaño del modelo |
|---|---|---|---|
| Reglas (ORB, homografía, Laplaciano, desviación estándar) | 0,905 | 0,07 s | 8,77 KB |
| CNN (Conv 32-64-128, densa 128, salida sigmoide) | 0,9975 | 0,02 s | 37,8 MB |

El artículo no evalúa la robustez ante cambios de iluminación ni ruido, y advierte que la CNN necesita datos
nuevos si la cámara cambia de posición. **La auditoría de robustez y la defensa son el aporte de este proyecto.**

## Tecnologías propuestas

| Tecnología | Uso en el proyecto | Por qué |
|---|---|---|
| Python 3.11 | Lenguaje principal de todo el flujo | Ecosistema maduro para visión artificial y aprendizaje profundo |
| OpenCV | Procesamiento de imágenes, Canny, ORB, homografía, Hough y cámara en vivo | Implementaciones rápidas de los algoritmos clásicos vistos en clase |
| scikit-image | Descriptores y utilidades de imagen | Complementa a OpenCV con funciones de análisis documentadas |
| NumPy | Operaciones con matrices de imagen | Base de todo el cálculo numérico en Python |
| pandas | Tablas de inventario y resultados | Facilita guardar y comparar métricas por condición |
| scikit-learn | Métricas: matriz de confusión, ROC/AUC, F1 | Funciones estándar y probadas para evaluar clasificadores |
| Matplotlib | Gráficas y figuras del informe | Estándar para curvas, diagramas de caja y matrices |
| TensorFlow/Keras con MobileNetV2 | ⏳ Módulo de aprendizaje profundo (propuesta, se confirma con el tema de CNN) | Transfer learning con un modelo liviano que converge rápido con pocos datos |
| pytest | Pruebas automáticas | Verifica cada módulo antes de integrarlo |
| Jupyter / Google Colab | Notebooks de análisis y GPU para entrenar | Permite documentar el análisis y entrenar sin GPU propia |
| MATLAB (Image Processing Toolbox y Computer Vision Toolbox) | Réplica de las perturbaciones y del módulo clásico; Deep Learning Toolbox opcional | La guía exige el proceso en Python y MATLAB |
| Git y GitHub | Control de versiones y Pull Requests | Trabajo en paralelo con revisión cruzada |

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
- [Plan de Cristhian](docs/PLAN_CRISTHIAN.md): datos, métricas, módulo clásico y auditoría.
- [Plan de Santiago](docs/PLAN_SANTIAGO.md): perturbaciones, MATLAB, CNN, captura y demo.

## Estado

Fase 0 (organización) lista: repositorio, estructura y plan. Siguiente paso: descarga y división de los datos.
