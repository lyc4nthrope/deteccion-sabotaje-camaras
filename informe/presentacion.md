# Presentación (T102): guion diapositiva por diapositiva

Esqueleto sobre `main`: cada diapositiva dice qué va, de qué sección del informe sale y qué se dice. La duración y el
formato (diapositivas o notebook) están por confirmar con la profesora; el guion tiene 15 diapositivas más 2 de
respaldo, y se recorta quitando las marcadas **(recortable)**. Todas las cifras salen de `results/`; no se copian a
mano. El reparto sugerido es C = Cristhian y S = Santiago.

| # | Diapositiva | Quién | Contenido | Sección |
|---|---|---|---|---|
| 1 | Portada | C | Título, grupo 7 · Seguridad, integrantes, docente | — |
| 2 | El problema | C | Cámaras tapadas, desenfocadas o giradas dejan de vigilar; cifras de Colombia y Armenia con su fuente | 1 |
| 3 | Qué se pregunta | C | No es "qué modelo acierta más", sino **cuál falla mejor y por qué**: auditar los dos enfoques ante perturbaciones y diagnosticar sus límites | 2 |
| 4 | Datos y rigor | C | Bremen, 751 duplicados descartados, división por bloques con banda de guarda, cámara no vista (ubicación 2) | 3 |
| 5 | Módulo 1: clásico | C | 8 descriptores, árbol de reglas de 4 líneas, umbrales calibrados solo con entrenamiento, referencias de la misma cámara | 5 |
| 6 | Módulo 2: CNN | S | MobileNetV2, base congelada, 5 124 parámetros entrenables, tasa 1e-2 elegida entre 7 corridas (1e-1 inestable) | 6 |
| 7 | Resultados limpios | S | Misma cámara: F1 0,954 (clásico) y 0,967 (CNN). **Cámara no vista: 0,920 frente a 0,580.** Tiempo: 12,6 frente a 81,8 ms | 7 |
| 8 | Auditoría de robustez | C | 6 perturbaciones × 3 niveles = 19 condiciones; la CNN gana en 10, el clásico en 4; el AUC se mantiene alto: fallan más en el **tipo** que en detectar | 8 |
| 9 | Diagnóstico del clásico | C | El ruido infla el Laplaciano: varianza ≈ 4,7 + 8,94 σ²; mediana ante sal y pimienta, gamma inversa ante gamma; CLAHE a ciegas daña (0,954 → 0,323) | 9.1–9.4 |
| 10 | Diagnóstico de la CNN | S | No hay descriptor que señalar, pero sí intervención: mediana antes de la red lleva sal y pimienta de 0,221 a 0,810; con JPEG no ayuda | 9.5 |
| 11 | **Demo en vivo** | S y C | Uno ataca la cámara y el otro explica y activa el filtro (`c`, `g`, `m`); ver `docs/ensayo_defensa.md` | `docs/demo.md` |
| 12 | Sistema híbrido (propuesta) | S | Clásico como detector principal, CNN como segunda opinión sobre el tipo; **no está medido**, y se dice | 12.2 |
| 13 | Limitaciones | C | Test mirado varias veces, sesgo de fechas por clase, sin captura física ni regional, causas de la CNN en su mayoría hipótesis | 10, 11 |
| 14 | Conclusiones | C y S | Ningún módulo es más robusto en todo; la prueba en otra cámara cambia la conclusión; el clásico se explica y se corrige | 12.1 |
| 15 | Trabajo futuro | S | Medellín (T76), captura en Armenia (T81, T75), Grad-CAM sobre imágenes degradadas, validar correcciones en validación | 12.2 |
| R1 | Respaldo: tabla ataque → causa → filtro | C | Tabla de `docs/defensa.md`, sección 3, para las preguntas | `docs/defensa.md` |
| R2 | Respaldo: Grad-CAM **(recortable)** | S | Atención difusa con sesgo central; no prueba que la red "mire el sabotaje" | 6.6 |

## Qué decir en las diapositivas que más pesan

**7 (resultados limpios).** "En la misma cámara parecen iguales. Solo la cámara que la red nunca vio muestra la
diferencia: el clásico compara contra referencias de *esa* cámara y generaliza; la CNN aprendió *su* cámara de
entrenamiento."

**9 y 10 (diagnóstico).** Es el 40 % de la nota. Decir siempre el orden **ataque → síntoma → causa → filtro → mejora
medida**, con el número. Reconocer lo que no se explica: JPEG y ruido gaussiano fuerte en la CNN siguen siendo
hipótesis, y ningún filtro probado pasa de F1 0,67 con σ ≥ 15 en el clásico.

**12 (híbrido).** Es una propuesta derivada de los datos, no un resultado. Decirlo en voz alta evita que lo
cuestionen como una afirmación sin medir.

## Lo que no se debe afirmar

- Que la CNN "mira el sabotaje" o "mira el contexto": Grad-CAM no lo demuestra (sección 6.6).
- Que el resultado de Medellín o de la captura propia generaliza: no se evaluaron (sección 12.1, objetivo 6).
- Que la mediana corrige todo: ante JPEG fuerte empeora y con ruido leve resta.

## Pendiente de confirmar

- Duración y formato de la presentación y de la defensa (con la profesora).
- Si se muestra o no la captura propia, según se haya hecho (T81).
- Las figuras: tomarlas de `results/figuras/` (`auditoria_f1_severidad.png`, `auditoria_matrices.png`,
  `auditoria_roc.png`, `descriptores_por_clase.png`, `gradcam_*.png`) y la curva de `results/cnn/lr1e-2_b32_frozen/curves.png`.
