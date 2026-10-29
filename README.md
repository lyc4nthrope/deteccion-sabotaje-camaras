# Detección de sabotaje en cámaras de vigilancia

Proyecto final de **Visión Artificial** · Grupo 7 · Temática: **Seguridad**

**Integrantes:** Cristhian Eduardo Osorio Restrepo y Santiago Orozco Zuluaga
**Docente:** Ing. Ana Maria Tamayo Ocampo, Mg

## Problema

Antes de un robo, es común que las cámaras de vigilancia sean **tapadas, desenfocadas o giradas**. Este proyecto
clasifica cada imagen de una cámara como **normal, desenfocada, girada o tapada** con dos enfoques en paralelo
(procesamiento clásico frente a aprendizaje profundo) y audita qué tan robustos son ante cambios de luz y ruido.

## Estructura

```
data/        Datos (no se suben al repositorio; ver data/README.md)
docs/        Plan del proyecto (docs/PLAN.md)
notebooks/   Notebooks de Python (compatibles con Colab)
src/         Código Python reutilizable
matlab/      Implementación en MATLAB
results/     Figuras y métricas generadas
informe/     Informe técnico y presentación
```

## Dataset

ComNets Camera Tampering Images Datasets, Universidad de Bremen: 3600 imágenes (1800 normales, 600 desenfocadas,
600 giradas y 600 con lente tapado).

> S. Attarha, K. Shanmugi y A. Förster, "Poster: Camera Tampering Detection for Outdoor IoT Systems",
> EWSN 2024. arXiv:2602.05706.

## Estado

Ver el plan y el avance por fases en [docs/PLAN.md](docs/PLAN.md).
