# 11. Transparencia y limitaciones

## 11.1 La prueba y la cámara no vista se evaluaron más de una vez

Los umbrales siempre se calibraron solo con el conjunto de entrenamiento, pero **las reglas cambiaron dos veces
después de ver resultados**. Por eso la prueba y la cámara no vista no se evaluaron una sola vez.

**Tabla 11.1.** Versiones de las reglas. Fuente: `docs/clasico.md`, sección 5.

| Versión | Qué cambió | F1 prueba | F1 cámara no vista | Giradas detectadas (no vista) |
|---|---|---|---|---|
| 1 | Reglas sin ángulo | 0,913 | 0,630 | 7 de 167 |
| 2 | Regla del ángulo, con umbral fijado de antemano en las normales de entrenamiento | 0,913 | 0,914 | 135 de 167 |
| 3 | Correcciones de una revisión de código: ángulo solo con 6 o más puntos RANSAC, umbral calculado con normales que coinciden, referencias fuera de la calibración | 0,954 | 0,920 | 141 de 167 |

- **Versión 2:** la regla del ángulo se agregó después de ver que, en la cámara no vista, "girada" es un giro
  **parcial** (la escena se superpone) y no un cambio completo de escena como en la ubicación 1.
- **Versión 3:** corrige defectos de código (ángulos de 150° calculados con 2 o 3 puntos inflaban el umbral), no
  ajustes a los resultados.

Los resultados de la prueba y de la cámara no vista son, por tanto, **optimistas**. Las pruebas que no se han
mirado durante el desarrollo son Medellín y la captura propia.

## 11.2 Las correcciones se compararon en el conjunto de prueba

El filtro de cada ataque se eligió por teoría antes de medir: mediana para el ruido impulsivo, gamma inversa para
la gamma, CLAHE para la oscuridad. Pero para el ruido gaussiano se probaron tres filtros (mediana, gaussiano y
bilateral) y se reporta el mejor, **medido en prueba**. Esa elección debería confirmarse en validación antes de
fijarla en la demo, igual que la decisión de dejar la mediana siempre activa.

## 11.3 Limitaciones de los datos

- **Clase girada pequeña:** 135 imágenes distintas en la ubicación 1 y solo 9 en prueba. Un error cambia su
  recall en 11 puntos; su evaluación principal es la cámara no vista (167 giradas).
- **Sesgo de fecha por clase:** cada clase se grabó en días distintos. Un modelo aprendido puede reconocer la fecha
  (luz, clima, vegetación) en lugar del sabotaje. Las reglas son menos sensibles porque comparan contra
  referencias de la misma cámara; los modelos aprendidos (sección 7.3) no.
- **Sin tapadas en la cámara no vista:** las de la ubicación 2 eran copias de la ubicación 1, así que la tapa
  solo se evalúa en la ubicación 1, en Medellín (sintética) y en la captura propia.
- **Sin tapas con la mano:** en Bremen las cámaras se taparon con objetos, así que `skin_ratio` no se pudo usar y
  la tapa con la mano no está medida. Se prueba con la captura propia.
- **Sabotaje sintético en Medellín:** el dataset no trae sabotaje, se genera por código; mide la generalización
  a otras cámaras, no la apariencia de un sabotaje real.
- **La auditoría usa solo la ubicación 1:** las perturbaciones se aplicaron a la prueba de la ubicación 1, no a la
  cámara no vista.

## 11.4 Límites del método clásico

- **Ruido gaussiano fuerte (σ ≥ 15):** ningún filtro probado recupera la regla de nitidez.
- **Sal y pimienta al 10 %:** la mediana de 3×3 solo recupera hasta F1 0,46.
- **Saturación por exceso de luz:** lo que llegó a 255 no se puede recuperar.
- **Imágenes sin color:** una imagen gris u oscura tiene `hue_std` = 0, igual que una tapa de un solo color; una
  cámara desenfocada de noche y sin color podría clasificarse como tapada.
- **Giros pequeños:** un giro menor de 0,94° es indistinguible de la vibración normal de la cámara.
- **Calibración por cámara:** el módulo necesita 8 imágenes de referencia de cada cámara nueva.
