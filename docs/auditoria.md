# Auditoría de robustez (T70, T71)

Código: `src/audit.py`. Resultados: `results/auditoria/tabla.csv` (métricas por condición) y
`results/auditoria/descriptores_test_degradado.csv` (descriptores de cada imagen degradada). Reproducir con
`python -m src.audit` (≈ 2 minutos).

## 1. Conjuntos degradados (T70)

- **Imágenes:** las 327 de **test** de la ubicación 1 (sin las descartadas).
- **Condiciones:** la original más 6 perturbaciones × 3 niveles (D7): brillo oscuro, brillo claro, gamma, ruido
  gaussiano, sal y pimienta y JPEG (`src/perturbations.py`). En total 19 condiciones y 6213 imágenes.
- **Orden:** cada imagen se reduce primero a 640×360 (el tamaño de los descriptores) y **después** se perturba.
  Si se perturbara la original (hasta 4608×2592) y luego se redujera, el ruido se promediaría y desaparecería.
- **Reproducibilidad:** las imágenes degradadas no se guardan. Cada una se regenera con una semilla fija
  (CRC32 de ruta, perturbación y nivel), así que la CNN puede reconstruir exactamente el mismo conjunto
  (`audit.degrade`).

## 2. Resultados del módulo clásico (T71)

La condición original reproduce exactamente el resultado de T43, lo que confirma que el procedimiento es coherente.

| Perturbación | Nivel | F1 macro | Falsas alarmas | AUC | ms por cuadro |
|---|---|---|---|---|---|
| original | — | 0,954 | 0,5 % | 0,997 | 15,0 |
| brillo oscuro | 1 / 2 / 3 | 0,955 / 0,935 / 0,842 | 1,6 % / 2,7 % / 2,7 % | 0,997 / 0,997 / 0,989 | 6,8–10,9 |
| brillo claro | 1 / 2 / 3 | 0,971 / 0,966 / 0,852 | 0,5 % / 1,1 % / 5,5 % | 0,997 / 0,997 / 0,994 | 12,3–12,8 |
| gamma | 1 / 2 / 3 | 0,902 / 0,762 / **0,599** | 1,1 % / 3,3 % / **18,7 %** | 0,997 / 0,995 / 0,978 | 10,3–12,8 |
| ruido gaussiano | 1 / 2 / 3 | **0,266 / 0,278 / 0,277** | 1,1 % / 0,5 % / 1,6 % | 0,997 / 0,997 / 0,997 | 12,5–19,6 |
| sal y pimienta | 1 / 2 / 3 | **0,278 / 0,267 / 0,232** | 0,5 % / 7,7 % / **27,5 %** | 0,997 / 0,992 / 0,964 | 14,5–20,6 |
| JPEG | 1 / 2 / 3 | 0,934 / 0,975 / 0,903 | 1,1 % / 0,5 % / 2,2 % | 0,997 / 0,997 / 0,995 | 11,5–11,9 |

## 3. Diagnóstico (para la defensa, T90)

**Lectura general:** con ruido, el AUC sigue en 0,997 y las falsas alarmas siguen bajas: el sistema sigue
sabiendo **que hay sabotaje**, pero se equivoca en **cuál** es.

| Ataque | Qué le pasa a los descriptores | Efecto en las reglas | Corrección propuesta (T92) |
|---|---|---|---|
| **Ruido gaussiano** (incluso σ = 5) | El ruido crea bordes falsos: el Laplaciano de una desenfocada sube de 4,7 a 230,9, por encima del umbral de 18,9 | Las 68 desenfocadas y las 68 tapadas pasan a "girada" (imagen "nítida" sin coincidencias) | Filtro de **mediana** o **bilateral** antes de medir la nitidez |
| **Sal y pimienta** | El Laplaciano de una desenfocada llega a 33 130 (nivel 3) | Igual que el ruido, y además 50 de 182 normales dan falsa alarma | Filtro de **mediana** (no lineal), la corrección clásica para este ruido |
| **Gamma** (nivel 3, escena muy oscura) | Pierde contraste: ORB encuentra menos coincidencias y los colores pierden saturación | 34 normales dan falsa alarma (18,7 %) y 42 de 68 tapadas pasan a desenfocadas | **Gamma inversa** del nivel o **CLAHE** |
| **Brillo claro** (nivel 3) | La saturación a blanco borra la textura | 10 normales pasan a "girada" | Inversa del brillo (PR #18); no recupera lo saturado |
| **Brillo oscuro y JPEG** | Cambios pequeños | El módulo resiste (F1 ≥ 0,84) | — |

Ya se había visto en el experimento de `docs/clasico.md` (sección 7): ningún suavizado gaussiano protege la regla
de nitidez del ruido. Por eso la corrección es no lineal.

## 4. Pendiente

- **CNN:** el mismo conjunto degradado, con `audit.degrade`, cuando se una el PR #11 (T64).
- **T72–T74:** matrices de confusión cruzadas, curvas ROC por condición y F1 frente a severidad, comparando ambos
  modelos.
- **T92:** aplicar las correcciones de la tabla anterior y medir el antes y el después.
