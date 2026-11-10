# 8. Auditoría de robustez

La guía pide enfrentar ambos sistemas a entornos degradados de forma artificial o física. Esta sección reporta la
degradación artificial del módulo clásico; la CNN, la degradación física y la prueba regional se agregan cuando
estén sus resultados.

## 8.1 Conjuntos degradados

- **Imágenes:** las 327 de prueba de la ubicación 1.
- **Perturbaciones:** 6 tipos con 3 niveles cada uno, más la original: 19 condiciones y 6213 imágenes. Los
  niveles van de leve a fuerte (tabla 8.1). Fuente: `src/perturbations.py`.
- **El desenfoque no es una perturbación:** es una clase. Una imagen normal desenfocada cambia de etiqueta con
  razón, así que no mide robustez.
- **Orden:** cada imagen se reduce primero a 640×360, el tamaño de los descriptores, y **después** se perturba.
  Si se perturbara la original (hasta 4608×2592) y luego se redujera, el ruido se promediaría y casi
  desaparecería.
- **Reproducibilidad:** las imágenes degradadas no se guardan. Cada una se regenera con una semilla fija (CRC32
  de la ruta, la perturbación y el nivel), así que la CNN se evalúa exactamente sobre las mismas imágenes
  (`src/audit.degrade`).

| Perturbación | Qué simula | Nivel 1 / 2 / 3 |
|---|---|---|
| Brillo oscuro | Anochecer o luz apagada (ganancia multiplicativa) | × 0,7 / 0,5 / 0,3 |
| Brillo claro | Contraluz o linterna | × 1,3 / 1,6 / 2,0 |
| Gamma | Sensor subexpuesto: oscurece los medios tonos | γ = 1,5 / 2 / 3 |
| Ruido gaussiano | Ruido del sensor con poca luz | σ = 5 / 15 / 30 niveles de gris |
| Sal y pimienta | Píxeles muertos o errores de transmisión | 1 % / 5 % / 10 % de los píxeles |
| JPEG | Compresión del video en la red | Calidad 60 / 30 / 10 |

**Tabla 8.1.** Perturbaciones y niveles.

## 8.2 Resultados del módulo clásico

**Tabla 8.2.** Módulo clásico en cada condición. Las predicciones son el número de imágenes asignadas a cada clase;
la distribución real es siempre 182 normales, 68 desenfocadas, 9 giradas y 68 tapadas. Fuente:
`results/auditoria/tabla.csv`.

| Perturbación | Nivel | Intensidad | F1 macro | Falsas alarmas | AUC | Predichas normal / desenfocada / girada / tapada |
|---|---|---|---|---|---|---|
| Original | — | — | 0,954 | 0,5 % | 0,997 | 183 / 68 / 8 / 68 |
| Brillo oscuro | 1 | × 0,7 | 0,955 | 1,6 % | 0,997 | 179 / 70 / 12 / 66 |
| Brillo oscuro | 2 | × 0,5 | 0,935 | 2,7 % | 0,997 | 177 / 70 / 14 / 66 |
| Brillo oscuro | 3 | × 0,3 | 0,842 | 2,7 % | 0,989 | 177 / 73 / 9 / 68 |
| Brillo claro | 1 | × 1,3 | 0,971 | 0,5 % | 0,997 | 182 / 68 / 9 / 68 |
| Brillo claro | 2 | × 1,6 | 0,966 | 1,1 % | 0,997 | 180 / 66 / 11 / 70 |
| Brillo claro | 3 | × 2,0 | 0,852 | 5,5 % | 0,994 | 172 / 66 / 19 / 70 |
| Gamma | 1 | γ = 1,5 | 0,902 | 1,1 % | 0,997 | 180 / 87 / 11 / 49 |
| Gamma | 2 | γ = 2 | 0,762 | 3,3 % | 0,995 | 176 / 108 / 14 / 29 |
| Gamma | 3 | γ = 3 | 0,599 | 18,7 % | 0,978 | 148 / 132 / 21 / 26 |
| Ruido gaussiano | 1 | σ = 5 | 0,266 | 1,1 % | 0,997 | 183 / 0 / 144 / 0 |
| Ruido gaussiano | 2 | σ = 15 | 0,278 | 0,5 % | 0,997 | 181 / 0 / 146 / 0 |
| Ruido gaussiano | 3 | σ = 30 | 0,277 | 1,6 % | 0,997 | 179 / 0 / 148 / 0 |
| Sal y pimienta | 1 | 1 % | 0,278 | 0,5 % | 0,997 | 181 / 0 / 146 / 0 |
| Sal y pimienta | 2 | 5 % | 0,267 | 7,7 % | 0,992 | 168 / 0 / 159 / 0 |
| Sal y pimienta | 3 | 10 % | 0,232 | 27,5 % | 0,964 | 132 / 0 / 195 / 0 |
| JPEG | 1 | calidad 60 | 0,934 | 1,1 % | 0,997 | 182 / 66 / 9 / 70 |
| JPEG | 2 | calidad 30 | 0,975 | 0,5 % | 0,997 | 181 / 67 / 10 / 69 |
| JPEG | 3 | calidad 10 | 0,903 | 2,2 % | 0,995 | 178 / 68 / 16 / 65 |

La condición original reproduce exactamente el resultado de prueba de la sección 7 (F1 0,954), lo que confirma
que el procedimiento de degradación no altera las imágenes cuando no hay perturbación.

## 8.3 Lectura

1. **El sistema sabe *que* hay sabotaje, pero no siempre *cuál*.** Con ruido gaussiano el F1 cae a 0,27, pero
   el AUC sigue en 0,997 y las falsas alarmas cerca del 1 %. Las normales siguen siendo normales; lo que falla
   es el tipo de sabotaje: **ninguna** imagen se predice desenfocada ni tapada, y todas pasan a girada (144 a 148
   predichas como giradas, con solo 9 giradas reales).
2. **El ruido impulsivo es el ataque más grave.** Sal y pimienta al 10 % combina los dos fallos: confunde el tipo
   de sabotaje (F1 0,23) y además dispara falsas alarmas en el 27,5 % de las normales.
3. **La oscuridad por gamma es el ataque de luz más grave.** Con γ = 3 las falsas alarmas suben al 18,7 % y las
   tapadas pasan a desenfocadas (26 predichas como tapadas de 68).
4. **El módulo resiste el brillo oscuro y la compresión JPEG.** El F1 se mantiene en 0,84 o más en todos los
   niveles, y el JPEG de calidad 10 todavía da 0,90.
5. **El tiempo por cuadro no cambia de forma relevante** (entre 6,8 y 20,6 ms según la condición): el costo está
   en los descriptores, no en la imagen.

El porqué de cada fallo, con su fórmula, y su corrección están en la sección 9.

## 8.4 Pendiente

- **CNN sobre las mismas 6213 imágenes** (Cristhian, T71, cuando se una la evaluación de la CNN), con matrices de
  confusión cruzadas modelo × condición crítica (T72), curvas ROC por condición y F1 frente a severidad (T73).
- **Degradación física** con luz real en la captura propia (Cristhian, T75, después de la captura T81).
- **Prueba regional en Medellín** y robustez de ORB, SIFT y KAZE ante perturbaciones (Santiago, T76 y T77).
