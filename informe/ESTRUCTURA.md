# Estructura del informe técnico (T100)

**Título:** Auditoría híbrida de modelos de visión para la detección de sabotaje en cámaras de vigilancia
**Grupo 7 · Seguridad** · Cristhian Eduardo Osorio Restrepo y Santiago Orozco Zuluaga
**Docente:** Ing. Ana Maria Tamayo Ocampo, Mg · **Entrega:** ≈ 15 de noviembre de 2026

Cada sección indica el criterio de la rúbrica que cubre, de dónde salen los datos (tarea y archivo del
repositorio) y quién la redacta (T101). La guía pone el énfasis en la **comparación entre metodologías y el
diagnóstico de sus límites**, no en una precisión alta en condiciones ideales: el informe debe reflejarlo.

| # | Sección | Criterio de la rúbrica | Fuente en el repositorio | Redacta |
|---|---|---|---|---|
| 1 | **Introducción y problema regional:** sabotaje de cámaras (tapar, desenfocar, girar) y su impacto en seguridad; cifras de Colombia y Armenia | Enfoque contextual | `docs/contexto_regional.md` (T06) | Cristhian |
| 2 | **Objetivos** general y específicos | — | Guía del proyecto, `README.md` | Cristhian |
| 3 | **Datos:** Bremen (con la nota aprobada por la profesora), hallazgos de calidad (duplicados, copias, fechas por clase), división por bloques sin fugas, cámara no vista, Medellín y captura en Armenia | Rigor (base de todo) | `docs/datos.md` (T11–T19), T16–T18, T80–T81 | Cristhian (Bremen), Santiago (Medellín, Armenia) |
| 4 | **Marco teórico breve:** Laplaciano, histograma y entropía, HSV, ORB/SIFT/KAZE, RANSAC, CNN y transfer learning, métricas | Fundamentos (R.A.1) | Clases 4, 9, 10, 11 y la de CNN | Ambos |
| 5 | **Módulo 1, clásico:** descriptores, reglas, umbrales calibrados y su justificación, puntaje continuo, comparación ORB/SIFT/KAZE | Enfoque clásico (20 %) | `docs/clasico.md` (T30–T43, T38), `results/figuras/descriptores_por_clase.png` | Cristhian |
| 6 | **Módulo 2, aprendizaje profundo:** arquitectura (MobileNetV2), hiperparámetros justificados, curvas de loss y accuracy, sobreajuste | Aprendizaje profundo (20 %) | `docs/cnn.md`, `results/cnn/` (T60–T66) | Santiago |
| 7 | **Resultados en condiciones limpias:** matrices de confusión, F1, falsas alarmas, ROC y AUC, tiempo por cuadro de ambos módulos en test y en la cámara no vista | Auditoría (20 %) | `results/clasico/metricas.csv`, T43, T64 | Cristhian (clásico), Santiago (CNN) |
| 8 | **Auditoría de robustez:** 6 perturbaciones × 3 niveles, matrices de confusión cruzadas, F1 frente a severidad, ROC por condición; degradación física (luz real); prueba regional en Medellín | Auditoría (20 %) | T70–T77, `results/auditoria/` | Cristhian (T70–T75), Santiago (T76–T77) |
| 9 | **Diagnóstico y correcciones:** por cada perturbación, qué descriptor o capa falla y por qué (teoría del píxel), filtro propuesto (CLAHE, gamma, mediana…) y mejora medida | **Defensa (40 %)** | T90–T93, `docs/defensa.md` | Ambos |
| 10 | **Discusión:** clásico frente a CNN (precisión, tiempo, datos necesarios, generalización a otra cámara); sesgo de fechas por clase; límites | Auditoría | Secciones 7 a 9 | Ambos |
| 11 | **Transparencia y limitaciones:** evaluaciones repetidas de test, clase girada con pocas imágenes, sabotaje sintético de Medellín | Rigor | `docs/clasico.md` (sección 5), `docs/datos.md` | Cristhian |
| 12 | **Conclusiones y trabajo futuro** | — | — | Ambos |
| 13 | **Referencias:** artículo de Bremen, datasets (Bremen, Medellín CC BY 4.0), fuentes regionales, OpenCV, Keras | — | `README.md`, `data/README.md`, `docs/contexto_regional.md` | Ambos |
| A | **Anexos:** cómo reproducir (comandos), registro de decisiones | — | `README.md`, `docs/DECISIONES.md` | Cristhian |

## Figuras y tablas previstas

| Figura o tabla | Sección | Fuente |
|---|---|---|
| Ejemplo de cada clase en las dos cámaras | 3 | T12 (galería) |
| Diagramas de caja de los descriptores por clase | 5 | `results/figuras/descriptores_por_clase.png` |
| Árbol de reglas y tabla de umbrales | 5 | `docs/clasico.md` |
| Curvas de loss y accuracy | 6 | T62 |
| Matrices de confusión (clásico y CNN; test y cámara no vista) | 7 | T43, T64 |
| Curvas ROC | 7 y 8 | `results/clasico/roc_*.csv`, T73 |
| F1 frente a severidad por perturbación (clásico frente a CNN) | 8 | T73 |
| Matrices de confusión cruzadas en condiciones críticas | 8 | T72 |
| Tabla de tiempos por cuadro | 7 | T43, T64 |
| Antes y después de cada corrección | 9 | T92 |

## Formato

- Pendiente de confirmar con la profesora el formato de entrega (documento o notebook) y la extensión.
- Tablas y figuras numeradas y citadas en el texto; cada número del informe debe salir de un archivo de
  `results/` (no se copian cifras a mano).
- Las imágenes con personas (Medellín, captura propia) no se publican sin difuminar (Ley 1581 de 2012).
