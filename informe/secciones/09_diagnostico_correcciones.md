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
| Brillo claro | 1 | Inversa del brillo del nivel | 0,971 | 0,954 | −0,017 | 0,5 % → 0,5 % |
| Brillo claro | 2 | Inversa del brillo del nivel | 0,966 | 0,946 | −0,020 | 1,1 % → 0,5 % |
| Brillo claro | 3 | Inversa del brillo del nivel | 0,852 | 0,795 | −0,058 | 5,5 % → 9,3 % |

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

6. **La inversa del brillo claro no corrige: empeora** (nivel 3: F1 0,852 → 0,795; falsas alarmas 5,5 % →
   9,3 %). El ataque multiplica el brillo y corta en 255: con × 2,0 queda saturado el 40,5 % de los píxeles,
   frente al 0,6 % de la imagen original (muestra fija de 40 imágenes de prueba,
   `results/auditoria/saturacion_brillo_claro.csv`). Dividir después no devuelve lo que se cortó: todo lo
   saturado queda en el mismo gris, a 21 niveles de gris en promedio de la imagen original. Es **pérdida de
   información**: ningún filtro posterior la recupera, y la defensa está en la captura (exposición automática).

## 9.5 Diagnóstico de la CNN

La CNN no expone descriptores: no se puede señalar "el descriptor que falla", como en el clásico. Lo que sí se
midió es **qué** pasa con cada ataque (síntoma y dirección del error, sobre las mismas imágenes que el clásico) y,
cuando una corrección aplicada **antes** de la red recupera la métrica, eso confirma la causa. Esa prueba por
intervención es el único método de esta sección que puede confirmar una causa: Grad-CAM se calculó solo sobre
imágenes sin degradar (sección 6.6) y sus mapas son centrales y difusos, así que **no confirma ni descarta
ninguna** de las causas de la tabla 9.5.

**Tabla 9.5. Diagnóstico de la CNN en el nivel 3 de cada ataque.** Fuentes: `results/auditoria/tabla.csv` y
`results/auditoria/matrices_cruzadas.csv`.

| Ataque | Síntoma medido | Hacia dónde se equivoca | Causa | Estado de la causa |
|---|---|---|---|---|
| Sal y pimienta (10 %) | F1 0,221; 33,5 % de falsas alarmas | 61 de 182 normales, las 68 desenfocadas y las 68 tapadas → girada | Los impulsos 0/255 aislados dominan los rasgos de las primeras capas y la escena deja de parecerse a la vista normal | **Confirmada** por la mediana (tabla 9.6) |
| JPEG (calidad 10) | F1 0,575; 0,5 % de falsas alarmas | 28 de 68 desenfocadas y 50 de 68 tapadas → girada | Los bloques de 8×8 crean bordes artificiales donde la imagen era lisa | Hipótesis: la mediana no la recupera |
| Ruido gaussiano (σ = 30) | F1 0,416; 1,1 % de falsas alarmas | 63 de 68 desenfocadas y 42 de 68 tapadas → girada | El ruido fuerte también parece textura y bordes | Hipótesis: la mediana ayuda a medias |
| Gamma (3) | F1 0,820; 0 falsas alarmas | 9 desenfocadas → girada; 5 giradas → normal | Los medios tonos se oscurecen, pero la estructura de la escena sigue | No probada con filtro |
| Brillo oscuro y claro (3) | F1 0,993 y 0,935 | Casi sin errores | La red se entrenó con escenas de distinta luz | No probada con filtro |

**Corrección antes de la red.** Se aplicó la mediana de 3×3 al cuadro degradado y después se redujo a la entrada de
la red, que es lo que haría la demo con el filtro activado (`src/audit_cnn_filter.py`,
`results/auditoria/cnn_con_filtro.csv`). Mismas 327 imágenes y mismas semillas que la auditoría.

**Tabla 9.6. F1 macro de la CNN sin y con mediana antes de la red.**

| Ataque | Nivel | F1 sin filtro | F1 con mediana | Diferencia | Falsas alarmas sin → con |
|---|---|---|---|---|---|
| Original | — | 0,967 | 0,967 | 0,000 | 0,0 % → 0,0 % |
| Sal y pimienta | 1 | 0,694 | 0,967 | +0,273 | 22,0 % → 0,0 % |
| Sal y pimienta | 2 | 0,244 | 0,967 | +0,724 | 25,3 % → 0,0 % |
| Sal y pimienta | 3 | 0,221 | 0,810 | +0,590 | 33,5 % → 0,0 % |
| JPEG | 1 | 0,948 | 0,967 | +0,019 | 0,0 % → 0,0 % |
| JPEG | 2 | 0,985 | 0,985 | 0,000 | 0,0 % → 0,0 % |
| JPEG | 3 | 0,575 | 0,522 | −0,052 | 0,5 % → 0,5 % |
| Ruido gaussiano | 1 | 0,985 | 0,967 | −0,017 | 0,0 % → 0,0 % |
| Ruido gaussiano | 2 | 0,764 | 0,878 | +0,114 | 0,5 % → 0,5 % |
| Ruido gaussiano | 3 | 0,416 | 0,470 | +0,055 | 1,1 % → 0,0 % |

**Lectura:**

1. **La mediana también corrige la sal y pimienta en la CNN, y mejor que en el clásico.** Con 10 % de impulsos
   devuelve la red de F1 0,221 a 0,810 y elimina las falsas alarmas; el clásico llega solo a 0,46 (sección 9.4).
   En los niveles 1 y 2 la red vuelve a su F1 sin degradar (0,967). Como quitar los impulsos devuelve el
   rendimiento, **los impulsos aislados son la causa del fallo**: no hace falta Grad-CAM para confirmarlo.
2. **No cuesta nada en imágenes limpias:** 0,967 antes y después. Dejarla siempre activa no daña a la CNN, como
   tampoco al clásico (sección 9.4), pero el efecto en otras condiciones se verificó solo en prueba.
3. **La mediana no sirve para todo.** Ante JPEG fuerte baja el F1 (0,575 → 0,522): el daño no es ruido aislado que
   el filtro pueda quitar, y la causa de los bloques de 8×8 queda como hipótesis. Con ruido gaussiano ayuda a
   medias (σ = 15: 0,764 → 0,878; σ = 30: 0,416 → 0,470) y con ruido leve resta (0,985 → 0,967). Un filtro
   aplicado a ciegas puede empeorar, como CLAHE en el clásico.
4. **Qué sí se puede decir de la CNN frente al clásico.** El clásico explica por qué falla (fórmula del Laplaciano)
   y tiene un filtro que lo corrige para cada ataque. La CNN no ofrece esa explicación, pero la prueba por
   intervención sí permite diagnosticar sus fallos: se coloca un filtro delante y se mide.

**Límites.** Una sola corrida de la CNN (semilla 0), la mediana de 3×3 como único filtro, y no se probó CLAHE ni el
bilateral delante de la red. Los valores de la sección 9.5 se calcularon sobre el conjunto de prueba, que ya se miró
más de una vez (sección 11).

## 9.6 Recomendaciones de preprocesamiento, calidad, formato y tamaño

Resumen de decisiones que salen de los resultados anteriores. Cada fila dice qué hacer, con qué número se sostiene y
qué tan firme es.

**Tabla 9.7.** Recomendaciones de preprocesamiento. Fuentes: tablas 9.4 y 9.6 y `results/auditoria/`.

| Decisión | Recomendación | Evidencia | Qué tan firme es |
|---|---|---|---|
| Tamaño de la imagen | Reducir a 640×360 para el clásico y a 224×224 para la CNN, como hacen los dos módulos. | Con ese tamaño se midieron todos los resultados; las imágenes originales llegan hasta 4608×2592 (sección 8.1). | **No medido:** no se probó otro tamaño, así que no se sabe si uno mayor o menor mejora o empeora. |
| Orden de los canales | Dar la imagen a la CNN en RGB y escalada de −1 a 1, no en BGR, que es el orden con que OpenCV lee. | `audit_cnn.to_cnn_input` invierte los canales antes de la red. | Es una condición del código, no un experimento. |
| Calidad JPEG | Guardar o transmitir con calidad 30 o más. | F1 de la CNN: 0,948 (calidad 60), 0,985 (30) y 0,575 (10). Clásico: 0,934, 0,975 y 0,903. | Medido en prueba, una corrida. Con calidad 10 la CNN falla y la mediana no la recupera (0,575 a 0,522). |
| Ruido de sal y pimienta | Mediana de 3×3 antes del módulo. | Niveles 1 y 2: clásico 0,278 a 0,982 y 0,267 a 0,986; CNN 0,694 a 0,967 y 0,244 a 0,967. Con 10 %: clásico 0,456 y CNN 0,810. | Medido. No daña las imágenes limpias (clásico 0,954 a 0,982; CNN 0,967 a 0,967). |
| Ruido gaussiano | Leve (σ = 5): filtro bilateral o gaussiano. Fuerte (σ de 15 en adelante): evitarlo en el origen, con más luz o menos ganancia del sensor. | Clásico con σ = 5: 0,266 a 0,966 con bilateral y a 0,862 con gaussiano. Con σ = 15 el mejor filtro llega a 0,665 y con σ = 30 ninguno pasa de 0,34. | Medido en prueba para el clásico. En la CNN la mediana ayuda a medias (σ = 15: 0,764 a 0,878). |
| Poca luz | Gamma inversa si se conoce cuánto se oscureció; no usar CLAHE a ciegas. | Gamma 3: 0,599 a 0,871. CLAHE sin ataque baja el F1 limpio de 0,954 a 0,323. | La gamma inversa usó el nivel real del ataque, así que 0,871 es una cota superior: un sistema real tendría que estimarlo. |
| Luz saturada | No hay filtro digital: bajar la exposición o cambiar el ángulo de la cámara. | La inversa del brillo empeora el nivel 3 (0,852 a 0,795). | Medido. Lo que se saturó en 255 no tiene información que recuperar. |

**Lo que no se midió:** el efecto del tamaño de la imagen, del formato sin pérdida (por ejemplo PNG) y de otros
códecs de video. Estas recomendaciones valen para el conjunto de prueba de la ubicación 1, que ya se miró más de una
vez (sección 11), y no se confirmaron en otra cámara.
