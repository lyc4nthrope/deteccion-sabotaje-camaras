# Anexo A. Cómo reproducir los resultados

Todo el código está en el repositorio. Los datos y los modelos no se versionan: se descargan y se regeneran.

## A.1 Entorno

- Python 3.13 y las dependencias de `requirements.txt`. OpenCV se fija en la versión 4 porque OpenCV 5 eliminó
  KAZE, que se usa en la comparación de detectores.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## A.2 Datos

Los 8 archivos zip del dataset de Bremen se descargan por WebDAV a `data/raw/bremen/zips/` y se descomprimen en
`data/raw/bremen/`. Las rutas exactas y el comando están en `data/README.md`, sección 1.

## A.3 Comandos, en orden

| Paso | Comando | Genera | Tiempo aproximado |
|---|---|---|---|
| 1 | `python -m src.data` | Inventario, división y referencias (`results/inventario.csv`, `data/splits/`) | Segundos |
| 2 | `python -m src.build_features` | Descriptores de todas las imágenes (`results/clasico/descriptores_*.csv`) | 2 minutos |
| 3 | `python -m src.evaluate_classic` | Umbrales, métricas, matrices, curvas ROC y tiempo (`results/clasico/`) | Segundos |
| 4 | `python -m src.compare_descriptors` | Comparación ORB, SIFT y KAZE (`results/clasico/comparacion_descriptores.csv`) | Minutos |
| 5 | `python -m src.ml_clasico` | KNN, SVM y Naive Bayes (`results/clasico/clasificadores.csv`) | Segundos |
| 6 | `python -m src.audit` | Auditoría de robustez (`results/auditoria/tabla.csv`) | 2 minutos |
| 7 | `python -m src.corrections_audit` | Correcciones antes y después (`results/auditoria/correcciones*.csv`) | 6 minutos |

Todos los pasos son deterministas: la división usa la semilla 42, las referencias se eligen repartidas por fecha,
sin azar, y cada imagen degradada usa una semilla fija (CRC32 de la ruta, la perturbación y el nivel). Volver a
correrlos produce las mismas métricas. Solo cambian los tiempos por cuadro (`tiempo.json` y la columna de
milisegundos), que dependen del equipo.

## A.4 Pruebas

```bash
pytest
```

Las pruebas cubren la división (sin imágenes repetidas entre conjuntos ni vecinas en el tiempo), los
descriptores, las reglas, las métricas, las perturbaciones, la auditoría y las correcciones. Las que necesitan el
dataset se omiten si no está descargado.

## A.5 Registro de decisiones

Cada decisión del proyecto, con las alternativas consideradas y su porqué, está en `docs/DECISIONES.md`.
