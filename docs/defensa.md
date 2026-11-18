# Guía de defensa: ataque → síntoma → causa → corrección (T90, T93)

La defensa (40 % de la nota) pide **diagnosticar con precisión matemática** por qué falla el modelo ante cambios de
luz o ruido en vivo, y **proponer el filtro exacto** que lo corrige. Esta guía reúne el diagnóstico del **módulo
clásico** (Cristhian); la sección de la **CNN** (Santiago) está en la sección 6.

Fuentes de los números: `docs/auditoria.md` (T70, T71, T92), `results/auditoria/` y `results/clasico/umbrales.json`.

## 1. Las reglas en una línea

```
coincidencias ORB ≥ 12,5  y |ángulo| ≤ 0,94°  → normal
coincidencias ORB ≥ 12,5  y (|ángulo| > 0,94° o sin ángulo confiable) → girada
coincidencias < 12,5 y Laplaciano < 18,9: dispersión del matiz < 5,4° → tapada; si no → desenfocada
coincidencias < 12,5 y Laplaciano ≥ 18,9 → girada
```

Primero se decide **si es la escena normal** con señales relativas a la cámara (coincidencias contra sus propias
referencias y ángulo); después, **qué sabotaje** es, con la nitidez y el color.

## 2. Dos fórmulas que explican casi todo

### 2.1 El ruido infla el Laplaciano

El Laplaciano de OpenCV (`cv2.Laplacian`, `ksize=1`) es el núcleo

```
 0  1  0
 1 -4  1
 0  1  0
```

Un filtro lineal aplicado a ruido independiente de varianza σ² produce ruido de varianza σ² × (suma de los pesos al
cuadrado) = σ² × (16 + 1 + 1 + 1 + 1) = **20 σ²**. Al pasar a gris (0,299 R + 0,587 G + 0,114 B, con ruido
independiente en cada canal), la desviación del ruido se reduce a σ × √(0,299² + 0,587² + 0,114²) = **0,6686 σ**.
Entonces, para una imagen desenfocada (nitidez propia ≈ 4,7):

**nitidez con ruido ≈ 4,7 + 20 × (0,6686 σ)² ≈ 4,7 + 8,94 σ²**

| σ del ruido | Predicho | Medido (mediana, desenfocadas de test) |
|---|---|---|
| 5 | 228 | 231 |
| 15 | 2016 | 2019 |
| 30 | 8050 | 8051 |

**Consecuencia:** una imagen desenfocada cruza el umbral de 18,9 con σ ≈ **1,26** niveles de gris, ruido casi
invisible: 4,7 + 8,94 × 1,26² ≈ 18,9. El ruido solo, sin contar la nitidez propia de la imagen, lo cruza con σ ≈ 1,45.
Desde ese punto una imagen desenfocada o tapada parece nítida.

### 2.2 El brillo escala el Laplaciano al cuadrado

El Laplaciano es lineal: si la imagen se multiplica por una ganancia g, el Laplaciano también, y su **varianza se
multiplica por g²**.

| Ganancia (brillo oscuro) | Predicho (831,7 × g²) | Medido (normales de test) |
|---|---|---|
| 0,7 | 408 | 409 |
| 0,5 | 208 | 210 |
| 0,3 | 75 | 77 |

Al aclarar (g > 1) la predicción deja de cumplirse porque los píxeles se saturan en 255: la saturación **borra
textura** en vez de multiplicarla.

## 3. Diagnóstico por ataque (módulo clásico)

Valores en mediana sobre las 327 imágenes de test; F1 macro y falsas alarmas (normales que dan alarma) de
`results/auditoria/tabla.csv`.

| Ataque | Síntoma | Causa (descriptor que se altera) | Corrección | Mejora medida |
|---|---|---|---|---|
| **Ruido gaussiano** (σ = 5, 15, 30) | F1 0,95 → 0,27; las 68 desenfocadas y las 68 tapadas salen "giradas". Falsas alarmas casi iguales (~1 %) y AUC 0,997: sigue detectando **que** hay sabotaje | La nitidez de una desenfocada sube de 4,7 a 231 / 2019 / 8051 (sección 2.1) y cruza el umbral de 18,9. Sin coincidencias y "nítida", la regla dice girada | **Bilateral** (suaviza sin borrar bordes) | σ = 5: 0,27 → **0,97**. σ = 15 y 30: ningún filtro probado pasa de 0,67: **límite del método** |
| **Sal y pimienta** (1 %, 5 %, 10 % de píxeles) | F1 0,28 / 0,27 / 0,23; falsas alarmas 0,5 / 7,7 / **27,5 %** | Cada impulso 0 o 255 genera un Laplaciano enorme: la nitidez de una desenfocada llega a 3311 / 33 130. En el nivel 3 además las coincidencias de las normales bajan de 214 a 61 | **Mediana** (no lineal): toma el valor central de la vecindad e ignora los extremos | 0,28 → **0,98** (niveles 1 y 2), 0,23 → 0,46 (nivel 3); falsas alarmas 27,5 % → **1,1 %** |
| **Gamma** (1,5 / 2 / 3) | F1 0,90 / 0,76 / 0,60; falsas alarmas hasta **18,7 %**; 42 de 68 tapadas pasan a desenfocadas | Oscurece los medios tonos: menos contraste, ORB encuentra menos puntos (coincidencias de las normales 214 → 128). En las tapadas, la dispersión del matiz sube de 0 a 6,2, sobre el umbral de 5,4 | **Gamma inversa del nivel** (exponente 1/g) | 0,60 → **0,87**; falsas alarmas 18,7 % → **1,1 %** |
| **Brillo oscuro** (× 0,7 / 0,5 / 0,3) | F1 0,96 / 0,94 / 0,84 | La nitidez baja con g² (sección 2.2) y las coincidencias bajan (214 → 56), pero siguen sobre 12,5: el clásico **resiste** | **CLAHE** solo en oscuridad extrema | Nivel 3: 0,84 → **0,95** |
| **Brillo claro** (× 1,3 / 1,6 / 2) | F1 0,97 / 0,97 / 0,85; falsas alarmas hasta 5,5 % | La saturación en 255 borra la textura: coincidencias de las normales 214 → 37; algunas normales salen "giradas" | **Inversa del brillo** (#18) | **No sirve: empeora.** Nivel 3: 0,85 → 0,80; falsas alarmas 5,5 → 9,3 %. Con × 2 se satura el 40,5 % de los píxeles y lo cortado en 255 no se recupera (`docs/auditoria.md`, 4.2, punto 6) |
| **JPEG** (calidad 60 / 30 / 10) | F1 0,93 / 0,98 / 0,90 | Cambios pequeños: el clásico resiste | Ninguna (el bilateral empeora un poco) | — |

## 4. Lo que NO hay que hacer: CLAHE a ciegas

Aplicado a imágenes **sin ataque**, CLAHE baja el F1 de 0,954 a **0,323**. Realza el contraste local también en las
imágenes borrosas y tapadas, que no tienen detalle real: la nitidez de una desenfocada pasa de 3,9 a 31,7 y la de
una tapada de 2,5 a 24,1, por encima de 18,9. El clásico las cree nítidas y las manda a "girada" (129 de 327).

**Regla para la demo (T85):** cada filtro solo ante su ataque. La **mediana** es la única segura de dejar activa
(sin ataque: 0,954 → 0,982), y esa decisión se debería validar con val antes de adoptarla.

## 5. Preguntas probables y cómo responderlas

| La profesora hace… | Qué pasa | Qué se responde |
|---|---|---|
| **Apaga la luz o tapa la lámpara** | Las coincidencias bajan pero el clásico casi siempre sigue diciendo "normal"; en oscuridad extrema puede dar alarma | "El Laplaciano escala con g² y ORB pierde puntos por bajo contraste, pero la decisión normal/saboteada usa las coincidencias contra las referencias de esta cámara. Si cae demasiado, CLAHE o gamma inversa lo recuperan (0,84 → 0,95)" |
| **Le pone una linterna o contraluz** | Algunas normales pasan a "girada" | "La saturación en 255 borra textura y ORB pierde coincidencias (214 → 37). Se corrige con la inversa del brillo, aunque lo saturado ya no se recupera" |
| **Agrega ruido digital** | Las desenfocadas y tapadas pasan a "girada" | "El ruido infla el Laplaciano: varianza ≈ 4,7 + 8,94 σ². Con σ = 5 da 228; medimos 231. Basta σ ≈ 1,3 para cruzar el umbral. Se corrige con bilateral o mediana antes de medir la nitidez" |
| **Tapa la cámara con la mano** | **No medido** (Bremen no tiene tapas con la mano; se prueba con la captura propia, T81) | Hipótesis: "Sin coincidencias y sin bordes; la piel es un solo tono, así que la dispersión del matiz es baja y sale tapada. Si la mano deja pasar luz con textura, podría salir desenfocada". Confirmarlo en el ensayo (T94) |
| **Gira o mueve la cámara** | Pocas coincidencias, o ángulo > 0,94° | "Si apunta a otra escena, las coincidencias contra las referencias caen; si el giro es parcial, RANSAC estima el ángulo y supera 0,94°, el percentil 99 de las normales" |
| **"¿Por qué no usaron solo la CNN?"** | — | "En test la CNN rinde igual (F1 0,967 frente a 0,954), pero en una cámara nunca vista cae a 0,58 y el clásico se mantiene en 0,92. Todo modelo que aprende de una sola cámara memoriza esa cámara; las reglas deciden con señales relativas a la cámara" |
| **"¿Cómo saben que no hay fuga de datos?"** | — | "Sin duplicados (MD5), división por bloques de 12 imágenes con 2 descartadas en cada frontera, umbrales solo con train, y la cámara no vista nunca se usó para elegir nada" |

## 6. CNN (T90 y T93)

**Lo medido** (Cristhian, T71–T73): síntoma y dirección del error de la CNN en el nivel 3 de cada ataque, sobre las
mismas imágenes que el clásico. Fuentes: `results/auditoria/tabla.csv` y `results/auditoria/matrices_cruzadas.csv`.
**Lo que falta** (Santiago, T67, PR #34): confirmar o descartar cada causa con Grad-CAM. Hasta entonces, la columna
"Causa probable" son **hipótesis** y se deben presentar así en la defensa.

| Ataque (nivel 3) | Síntoma medido | Hacia dónde se equivoca | Causa probable (hipótesis) | Frente al clásico |
|---|---|---|---|---|
| **Sal y pimienta** (10 %) | F1 0,221; **33,5 %** de falsas alarmas; AUC 0,872 | 61 de 182 normales, las 68 desenfocadas y las 68 tapadas → girada | Los impulsos 0/255 dominan los rasgos de bordes de las primeras capas y la escena deja de parecerse a la vista normal | Los dos fallan, pero el clásico se recupera con la **mediana** (0,23 → 0,46; niveles 1 y 2: 0,98) y la CNN no tiene un filtro equivalente medido |
| **JPEG** (calidad 10) | F1 0,575; falsas alarmas 0,5 % | 28 de 68 desenfocadas y 50 de 68 tapadas → girada | Los bloques de 8×8 de la compresión crean bordes artificiales donde la imagen era lisa (desenfocada, tapada) | **El clásico resiste (0,903)**; por qué resiste no está medido |
| **Ruido gaussiano** (σ = 30) | F1 0,416; falsas alarmas 1,1 % | 63 de 68 desenfocadas y 42 de 68 tapadas → girada | El ruido fuerte también parece textura y bordes | La CNN aguanta más: σ = 5 da 0,985 frente a 0,266 del clásico, cuyo Laplaciano se infla desde el nivel 1 (sección 2.1) |
| **Gamma** (3) | F1 0,820; 0 falsas alarmas | 9 desenfocadas → girada; 5 giradas → normal | Los medios tonos se oscurecen, pero la estructura de la escena sigue | **La CNN resiste mejor** (el clásico: 0,599 y 18,7 % de falsas alarmas) |
| **Brillo oscuro y claro** (3) | F1 0,993 y 0,935 | Casi sin errores | La red se entrenó con escenas de distinta luz | La CNN resiste mejor; el clásico baja a 0,842 y 0,852 |

**Cómo decirlo en la defensa:** "La CNN no es más robusta en todo. Gana con la luz y el ruido leve; pierde con la
compresión fuerte, y con sal y pimienta da muchas falsas alarmas. Y cuando falla, no podemos señalar un descriptor ni
corregirlo con un filtro conocido, como sí hacemos con el clásico (mediana ante impulsos, gamma inversa ante la
oscuridad). Una propuesta coherente con estos datos es un sistema híbrido: el clásico como detector principal de
normal frente a saboteada, porque generaliza a una cámara nueva con 8 referencias (F1 0,920 frente a 0,580 de la CNN
en la cámara no vista), y la CNN como segunda opinión sobre el tipo en cámaras con imágenes de entrenamiento, donde
resiste mejor la luz y el ruido leve. Es una propuesta: no la medimos." (Informe, sección 12.2.)

**Pendiente para Santiago:** con los mapas de Grad-CAM (#34), confirmar o descartar las hipótesis de sal y pimienta y
JPEG, y medir si aplicar la mediana **antes** de la CNN recupera la sal y pimienta (hoy solo está medido para el
clásico).

## 7. Límites que conviene reconocer

- Ruido gaussiano fuerte (σ ≥ 15): ningún filtro probado recupera la regla de nitidez.
- Sal y pimienta al 10 %: la mediana de 3×3 solo recupera hasta F1 0,46.
- Lo saturado por exceso de luz no se puede recuperar.
- Una imagen gris u oscura sin color tiene dispersión del matiz 0, igual que una tapa (ver `docs/clasico.md`).
- Los resultados de test se miraron más de una vez durante el desarrollo (`docs/clasico.md`, sección 5).
