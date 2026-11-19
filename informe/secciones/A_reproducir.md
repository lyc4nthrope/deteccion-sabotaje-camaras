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
`data/raw/bremen/`. Las rutas exactas y el comando están en `data/README.md`, sección 1. Las 6324 imágenes de Medellín van en
`data/raw/medellin/` (`data/README.md`, sección 2).

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
| 8 | `python -m src.train_cnn --name lr1e-2_b32_frozen --lr 1e-2 --batch 32 --seed 0` | Modelo y curvas de la CNN (`results/cnn/lr1e-2_b32_frozen/`) | 3 minutos en CPU |
| 9 | `python -m src.evaluate_cnn` | Métricas, matrices y tiempo de la CNN (`results/cnn/lr1e-2_b32_frozen/evaluacion/`) | Minutos |
| 10 | `python -m src.compare_architectures` | Comparación de arquitecturas (`results/cnn/arquitecturas.csv`) | Minutos |
| 11 | `python -m src.explain_cnn` | Mapas Grad-CAM (`results/figuras/gradcam_*.png`) | Minutos |
| 12 | `python -m src.audit_cnn --run lr1e-2_b32_frozen` | CNN en las condiciones degradadas (`results/auditoria/predicciones_cnn_test_degradado.csv`) | Minutos |
| 13 | `python -m src.audit_cnn_filter --run lr1e-2_b32_frozen` | Mediana antes de la CNN (`results/auditoria/cnn_con_filtro.csv`) | Minutos |
| 14 | `python -m src.audit_figures` | Figuras de la auditoría (`results/figuras/auditoria_*.png`) | Minutos |
| 15 | `python -m src.medellin_groups` | Grupos de cámara de Medellín (`data/splits/medellin_grupos.csv`) | 15 minutos |
| 16 | `python -m src.medellin` | Las 600 imágenes normales (`data/splits/medellin_normales.csv`) | Segundos |
| 17 | `python -m src.synthetic_tampering --list data/splits/medellin_normales.csv --out data/medellin_sintetico` | Sabotaje sintético (no se versiona) | Minutos |
| 18 | `python -m src.evaluate_medellin` | Prueba cruzada Bremen a Medellín (`results/medellin/`) | Minutos |
| 19 | `python informe/build_docx.py` | Este informe en Word con la plantilla de la universidad (necesita `markdown-it-py`) | Segundos |

Todos los pasos son deterministas: la división usa la semilla 42, las referencias se eligen repartidas por fecha,
sin azar, y cada imagen degradada usa una semilla fija (CRC32 de la ruta, la perturbación y el nivel). Volver a
correrlos produce las mismas métricas. La CNN usa la semilla 0 y reproduce las métricas del informe. Solo cambian los tiempos por cuadro (`tiempo.json` y la columna de
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
