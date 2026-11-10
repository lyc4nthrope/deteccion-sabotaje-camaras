# 9. Diagnóstico y correcciones

La defensa pide diagnosticar con precisión matemática por qué falla el modelo ante cambios de luz o ruido y
proponer el filtro exacto que lo corrige. Esta sección cubre el módulo clásico; la parte de la CNN la redacta
Santiago (sección 9.5).

## 9.1 Por qué el ruido rompe la regla de nitidez

El Laplaciano de OpenCV (`cv2.Laplacian`, `ksize=1`) es el núcleo

```
 0  1  0
 1 -4  1
 0  1  0
```

Un filtro lineal aplicado a ruido independiente de varianza σ² produce ruido de varianza σ² multiplicada por la
suma de los pesos al cuadrado: σ² × (16 + 1 + 1 + 1 + 1) = **20 σ²**. Al pasar a gris (0,299 R + 0,587 G +
0,114 B, con ruido independiente en cada canal), la desviación del ruido se reduce a
σ × √(0,299² + 0,587² + 0,114²) = 0,6686 σ. Para una imagen desenfocada, con nitidez propia de 4,7:

  **nitidez con ruido ≈ 4,7 + 20 × (0,6686 σ)² = 4,7 + 8,94 σ²**

**Tabla 9.1.** Predicción frente a medición (mediana de las desenfocadas de prueba). Fuente: `docs/defensa.md`,
sección 2.1.

| σ del ruido | Predicho | Medido |
|---|---|---|
| 5 | 228 | 231 |
| 15 | 2016 | 2019 |
| 30 | 8050 | 8051 |

La fórmula predice la medición con menos de 2 % de error. Una imagen desenfocada cruza el umbral de 18,9 con
σ ≈ 1,26 niveles de gris (4,7 + 8,94 × 1,26² ≈ 18,9), un ruido casi invisible. Desde ahí, una imagen sin bordes
parece nítida y, sin coincidencias ORB, la regla la declara girada. Es exactamente el patrón de la tabla 8.2.

## 9.2 Por qué el brillo cambia la nitidez

El Laplaciano es lineal: si la imagen se multiplica por una ganancia g, el Laplaciano también, y su **varianza se
multiplica por g²**. Con las normales de prueba (nitidez mediana de 831,7):

| Ganancia | Predicho (831,7 × g²) | Medido |
|---|---|---|
| 0,7 | 408 | 409 |
| 0,5 | 208 | 210 |
| 0,3 | 75 | 77 |

**Tabla 9.2.** Fuente: `docs/defensa.md`, sección 2.2.

Incluso con g = 0,3 la nitidez de una normal (75) queda muy por encima del umbral (18,9), y por eso el brillo
oscuro casi no afecta al módulo. Al aclarar (g > 1) la predicción deja de cumplirse, porque los píxeles se saturan
en 255: la saturación **borra textura** en vez de multiplicarla.

## 9.3 Diagnóstico por ataque

| Ataque | Descriptor que se altera | Efecto en las reglas | Corrección |
|---|---|---|---|
| Ruido gaussiano | `lap_var` sube con 8,94 σ² (sección 9.1) | Desenfocadas y tapadas pasan a girada | **Bilateral** o mediana antes de medir la nitidez |
| Sal y pimienta | Cada impulso 0 o 255 da un Laplaciano enorme; en el nivel 3 las coincidencias ORB de las normales bajan de 214 a 61 | Igual que el ruido, más falsas alarmas | **Mediana**: toma el valor central de la vecindad e ignora los extremos |
| Gamma | Menos contraste: las coincidencias de las normales bajan de 214 a 128; en las tapadas, `hue_std` sube de 0 a 6,2, sobre el umbral de 5,4 | Falsas alarmas; tapadas como desenfocadas | **Gamma inversa** del nivel (exponente 1/γ) |
| Brillo oscuro | `lap_var` baja con g²; coincidencias de 214 a 56, aún sobre 12,5 | Resiste | **CLAHE**, solo en oscuridad extrema |
| Brillo claro | La saturación borra textura: coincidencias de 214 a 37 | Algunas normales pasan a girada | Inversa del brillo; lo saturado no se recupera |
| JPEG | Cambios pequeños | Resiste | Ninguna |

**Tabla 9.3.** Valores en mediana sobre las 327 imágenes de prueba. Fuente: `docs/defensa.md`, sección 3.

## 9.4 Correcciones: antes y después

Cada filtro se aplica después de la degradación y antes de los descriptores (`src/corrections_audit.py`). Además,
cada filtro se prueba en las imágenes **limpias** (filas "Original"): en la demo no se sabe de antemano qué ataque
hay, así que un filtro no debe dañar las imágenes normales.

**Tabla 9.4.** F1 macro y falsas alarmas antes y después de cada corrección. Fuente:
`results/auditoria/correcciones_antes_despues.csv`.

| Perturbación | Nivel | Corrección | F1 antes | F1 después | Cambio | Falsas alarmas |
|---|---|---|---|---|---|---|
| Original | — | CLAHE | 0,954 | 0,323 | −0,631 | 0,5 % → 0,0 % |
| Brillo oscuro | 1 | CLAHE | 0,955 | 0,489 | −0,466 | 1,6 % → 0,5 % |
| Brillo oscuro | 2 | CLAHE | 0,935 | 0,636 | −0,299 | 2,7 % → 0,0 % |
| Brillo oscuro | 3 | CLAHE | 0,842 | 0,953 | +0,111 | 2,7 % → 0,5 % |
| Gamma | 1 | CLAHE | 0,902 | 0,301 | −0,601 | 1,1 % → 1,1 % |
| Gamma | 2 | CLAHE | 0,762 | 0,321 | −0,441 | 3,3 % → 2,2 % |
| Gamma | 3 | CLAHE | 0,599 | 0,347 | −0,252 | 18,7 % → 3,3 % |
| Original | — | Mediana | 0,954 | 0,982 | +0,029 | 0,5 % → 0,5 % |
| Ruido gaussiano | 1 | Mediana | 0,266 | 0,783 | +0,516 | 1,1 % → 1,1 % |
| Ruido gaussiano | 2 | Mediana | 0,278 | 0,277 | −0,001 | 0,5 % → 1,1 % |
| Ruido gaussiano | 3 | Mediana | 0,277 | 0,277 | +0,001 | 1,6 % → 1,1 % |
| Sal y pimienta | 1 | Mediana | 0,278 | 0,982 | +0,704 | 0,5 % → 0,5 % |
| Sal y pimienta | 2 | Mediana | 0,267 | 0,986 | +0,719 | 7,7 % → 0,5 % |
| Sal y pimienta | 3 | Mediana | 0,232 | 0,456 | +0,223 | 27,5 % → 1,1 % |
| Original | — | Gaussiano | 0,954 | 0,876 | −0,078 | 0,5 % → 0,5 % |
| Ruido gaussiano | 1 | Gaussiano | 0,266 | 0,862 | +0,596 | 1,1 % → 0,5 % |
| Ruido gaussiano | 2 | Gaussiano | 0,278 | 0,665 | +0,386 | 0,5 % → 1,1 % |
| Ruido gaussiano | 3 | Gaussiano | 0,277 | 0,331 | +0,054 | 1,6 % → 1,6 % |
| Original | — | Bilateral | 0,954 | 0,941 | −0,013 | 0,5 % → 0,5 % |
| Ruido gaussiano | 1 | Bilateral | 0,266 | 0,966 | +0,700 | 1,1 % → 1,1 % |
| Ruido gaussiano | 2 | Bilateral | 0,278 | 0,386 | +0,108 | 0,5 % → 1,1 % |
| Ruido gaussiano | 3 | Bilateral | 0,277 | 0,277 | +0,001 | 1,6 % → 1,1 % |
| JPEG | 1 | Bilateral | 0,934 | 0,905 | −0,029 | 1,1 % → 1,1 % |
| JPEG | 2 | Bilateral | 0,975 | 0,845 | −0,130 | 0,5 % → 2,2 % |
| JPEG | 3 | Bilateral | 0,903 | 0,838 | −0,065 | 2,2 % → 2,2 % |
| Gamma | 1 | Gamma inversa del nivel | 0,902 | 0,954 | +0,051 | 1,1 % → 0,5 % |
| Gamma | 2 | Gamma inversa del nivel | 0,762 | 0,844 | +0,082 | 3,3 % → 1,1 % |
| Gamma | 3 | Gamma inversa del nivel | 0,599 | 0,871 | +0,272 | 18,7 % → 1,1 % |

**Lectura:**

1. **La mediana es la corrección del ruido impulsivo.** Recupera la sal y pimienta de niveles 1 y 2 (F1 0,98) y
   baja las falsas alarmas del nivel 3 de 27,5 % a 1,1 %. Es la respuesta no lineal que predice la teoría: el
   valor central de la vecindad ignora los píxeles extremos. Con 10 % de impulsos, la mediana de 3×3 ya no
   alcanza a recuperar el tipo de sabotaje (F1 0,46).
2. **Para el ruido gaussiano, el bilateral funciona solo con ruido leve** (σ = 5: 0,27 → 0,97). Con σ = 15 y 30
   ningún filtro probado pasa de 0,67: el ruido fuerte se confunde con los bordes reales. Es un **límite del
   método clásico** basado en el Laplaciano.
3. **La gamma inversa del nivel corrige la oscuridad por potencia**: con γ = 3, el F1 sube de 0,60 a 0,87 y las
   falsas alarmas bajan de 18,7 % a 1,1 %.
4. **CLAHE es peligroso si se aplica a ciegas.** En imágenes limpias baja el F1 de 0,954 a 0,323: realza el
   contraste local también en las imágenes borrosas y tapadas, cuya nitidez sube por encima del umbral, y el
   clásico las manda a girada. Solo ayuda en la oscuridad extrema (brillo oscuro nivel 3: 0,84 → 0,95). Una
   corrección aplicada sin diagnóstico puede hacer más daño que el ataque.
5. **Consecuencia para la demo:** cada filtro se activa solo ante su ataque: mediana ante ruido impulsivo,
   bilateral ante ruido leve, gamma inversa o CLAHE ante oscuridad. La mediana es la única que no daña las
   imágenes limpias (0,954 → 0,982), pero dejarla siempre activa es una decisión que se debe validar con el
   conjunto de validación, no con el de prueba (sección 11).

La corrección del brillo claro (inversa del brillo) se agrega cuando se una su implementación.

## 9.5 Diagnóstico de la CNN

*Redacta Santiago (T90 y T93): mismo formato (ataque → descriptor o capa que falla → causa → corrección → mejora)
con la CNN sobre el mismo conjunto degradado y la explicabilidad con Grad-CAM (T67).*
