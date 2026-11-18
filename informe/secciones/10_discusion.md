# 10. Discusión

La guía pide comparar las dos metodologías y diagnosticar sus límites. Esta sección reúne los resultados de las
secciones 7 a 9 y responde qué módulo conviene en cada situación y por qué.

## 10.1 Resumen de la comparación

**Tabla 10.1.** Clásico frente a CNN. Fuentes: `results/clasico/metricas.csv`, `results/clasico/tiempo.json`,
`results/cnn/lr1e-2_b32_frozen/evaluacion/resumen.csv`, `results/cnn/lr1e-2_b32_frozen/evaluacion/tiempo.json`,
`results/cnn/lr1e-2_b32_frozen/run.json` y `results/auditoria/tabla.csv`.

| Criterio | Clásico | CNN | Ventaja |
|---|---|---|---|
| F1 macro en prueba (327 imágenes) | 0,954 | 0,967 | Empate práctico |
| Falsas alarmas en prueba | 0,5 % | 0,0 % | Empate práctico (1 de 182 normales) |
| AUC binaria en prueba | 0,997 | 1,000 | Empate práctico |
| F1 macro en la cámara no vista (1163 imágenes) | **0,920** | 0,580 | **Clásico** |
| Falsas alarmas en la cámara no vista | **0,1 %** | 4,2 % | **Clásico** |
| AUC binaria en la cámara no vista | **0,979** | 0,753 | **Clásico** |
| Auditoría: condiciones con mejor F1 (de 19) | 4 | **10** | **CNN** (5 empates) |
| Tiempo por cuadro, mediana de las 19 condiciones de la auditoría | **12,4 ms** | 51,9 ms | **Clásico** (unas 4 veces) |
| Qué se ajusta con datos | 5 parámetros de decisión (tabla 5.1), más 8 referencias por cámara | 5 124 pesos sobre una base preentrenada | Clásico (menos datos) |
| Explicación y corrección de cada fallo | Fórmula y filtro medidos (sección 9) | Hipótesis, Grad-CAM pendiente | **Clásico** |

La respuesta corta es que **ningún módulo es mejor en todo**: en condiciones limpias empatan, ante perturbaciones de
la misma cámara gana la CNN, y en una cámara nueva gana con amplitud el clásico.

## 10.2 Precisión y falsas alarmas en condiciones limpias

En la prueba de la ubicación 1 los dos módulos rinden casi igual (F1 0,954 frente a 0,967). Las matrices de
confusión muestran que la diferencia está en una sola imagen: los dos aciertan las 68 desenfocadas y las 68
tapadas y confunden 2 de las 9 giradas con normales; el clásico, además, marca 1 normal como girada y la CNN
ninguna (`results/clasico/matriz_test.csv` y `results/cnn/lr1e-2_b32_frozen/evaluacion/matriz_test.csv`). Con 9
giradas, un error mueve el recall de esa clase en 11 puntos (sección 7.1), así que esta diferencia no permite
elegir un módulo.

Las dos cifras de prueba son además **optimistas**: la prueba del clásico se evaluó varias veces mientras cambiaban
las reglas (sección 11.1), y la validación de la CNN casi no discrimina entre configuraciones, porque comparte
cámara y días con el entrenamiento (`docs/cnn.md`, T66). La prueba de la ubicación 1 mide, sobre todo, si el
modelo reconoce esa cámara en esos días.

## 10.3 Generalización a una cámara no vista

Es el resultado más importante del proyecto. En la ubicación 2, que ningún módulo vio al entrenar ni al calibrar:

- El **clásico** baja de F1 0,954 a 0,920 y mantiene las falsas alarmas en 0,1 % (1 de 745 normales).
- La **CNN** baja de 0,967 a 0,580, con 4,2 % de falsas alarmas y AUC de 0,753. Envía a "normal" 110 de las 251
  desenfocadas y 102 de las 167 giradas; solo acierta 80 desenfocadas y 65 giradas
  (`results/cnn/lr1e-2_b32_frozen/evaluacion/matriz_camara_no_vista.csv`). El clásico acierta 216 y 141
  (`results/clasico/matriz_camara_no_vista.csv`).

La sección 7.3 da la clave para interpretar esta diferencia: KNN, SVM y Naive Bayes entrenados **sobre los mismos
descriptores del clásico** también se derrumban en la cámara no vista (F1 entre 0,34 y 0,45). Por tanto, la
diferencia no está entre "clásico" y "profundo", sino entre **decidir con señales relativas a la cámara** y
**aprender valores absolutos de una sola cámara**. Las reglas preguntan primero si la imagen coincide con 8
referencias de la misma cámara (ORB y ángulo); cualquier modelo que aprende la apariencia de la ubicación 1
—umbrales sobre descriptores absolutos o pesos de una CNN— aprende también su escena, su luz y su contraste, y esos
valores cambian en otra cámara (una desenfocada de la cámara 2 tiene el contraste de una normal de la cámara 1,
sección 7.3).

El intento de corregirlo con aumentación (volteo, traslación, contraste y brillo) y dropout 0,5 empeoró la cámara no
vista a F1 0,333 (`results/cnn/lr1e-2_b32_frozen_aug_drop05/evaluacion/resumen.csv`). Es coherente con lo anterior:
una aumentación fotométrica simula otra luz en **la misma** escena, no otra escena. No se probaron más variantes
para no seguir ajustando sobre la cámara no vista (`docs/cnn.md`, T63).

El precio del clásico es la **calibración**: necesita 8 imágenes normales de cada cámara nueva antes de funcionar
(sección 3.4). Es un paso barato al instalar una cámara, pero es una dependencia real: sin referencias, la regla
principal no existe.

## 10.4 Sesgo de fechas por clase

En la ubicación 1 cada clase se grabó en días distintos: normales del 16 al 18 de julio de 2024, desenfocadas del
18 al 20, tapadas del 22 y 23 de julio y giradas del 24 y 25 de agosto (sección 3.2). La división por bloques
evita que dos fotos casi iguales queden a ambos lados, pero **no puede** separar los días: dentro de cada clase, el
entrenamiento, la validación y la prueba salen de los mismos días. Por eso la prueba de la ubicación 1 no detecta
este sesgo; un modelo que aprenda "luz de agosto, por tanto girada" acierta igual en prueba.

La caída de la CNN en la cámara no vista es **compatible** con ese atajo, pero no lo demuestra: también la explican
el cambio de escena, de resolución (4608×2592 frente a 1280×720) y de tipo de giro. Este último punto es concreto:
en la ubicación 1 "girada" es un cambio completo de escena, mientras que en la ubicación 2 es un giro parcial con la
escena superpuesta (sección 11.1). Que 102 de 167 giradas salgan como normales es lo esperable si la red aprendió
"otra escena" y no "giro"; es una **hipótesis**. Distinguir entre fecha, escena y tipo de giro requiere mirar
dónde se fija la red, con Grad-CAM (T67), que está pendiente.

El clásico es **menos sensible** al sesgo por construcción, no por medición: sus parámetros de decisión (tabla 5.1) tienen un
significado físico fijado de antemano (nitidez, dispersión del matiz, coincidencias, ángulo) y la decisión principal se compara
contra referencias de la misma cámara. Aun así, sus umbrales se calibraron con esos mismos días, y la auditoría
física con luz real (T75) sigue pendiente.

## 10.5 Robustez ante perturbaciones

La auditoría (sección 8) aplica las mismas 6 perturbaciones × 3 niveles a las mismas 327 imágenes de prueba, con la
misma semilla por imagen para los dos módulos. La CNN tiene mejor F1 en 10 de 19 condiciones, el clásico en 4 y en
5 empatan (diferencia de 0,02 o menos).

**Dónde gana la CNN y por qué.** Con la luz (gamma 3: 0,820 frente a 0,599; brillo oscuro 3: 0,993 frente a 0,842)
y con el ruido gaussiano leve (σ = 5: 0,985 frente a 0,266). El fallo del clásico tiene una causa exacta: el
Laplaciano de una imagen desenfocada sube como 4,7 + 8,94 σ² y cruza el umbral de 18,9 con σ ≈ 1,26 niveles de gris
(sección 9.1); desde ahí una imagen sin bordes parece nítida y sale "girada". La CNN no depende de un solo
descriptor con un umbral fijo, así que un ruido pequeño no la desplaza. Que la red resista mejor la luz es coherente
con que vio escenas de día y de noche al entrenar, pero esa causa es una **hipótesis** (`docs/defensa.md`,
sección 6).

**Dónde gana el clásico.** Con la compresión JPEG fuerte (calidad 10: 0,903 frente a 0,575) y con el brillo claro
leve (niveles 1 y 2). Con JPEG 3 la CNN envía a "girada" 28 de 68 desenfocadas y 50 de 68 tapadas (sección 8.5). La
hipótesis es que los bloques de 8×8 de la compresión crean bordes artificiales justo donde la imagen era lisa; por
qué el clásico resiste ese mismo ataque no está medido.

**Dónde fallan los dos.** Con sal y pimienta al 10 % (F1 0,232 y 0,221), y con falsas alarmas: 27,5 % el clásico y
33,5 % la CNN. Con ruido gaussiano fuerte (σ = 30) la CNN también cae, a 0,416.

**Qué pasa después del fallo.** Aquí está la diferencia que más pesa para la defensa. El fallo del clásico se
explica con una fórmula que predice la medición con menos de 2 % de error (tabla 9.1) y se corrige con un filtro
elegido por teoría: la mediana lleva la sal y pimienta de niveles 1 y 2 a F1 0,98 y baja las falsas alarmas del
nivel 3 de 27,5 % a 1,1 %; la gamma inversa lleva gamma 3 de 0,599 a 0,871 (tabla 9.4). Para la CNN no hay todavía
ni diagnóstico confirmado ni corrección medida: si la mediana aplicada **antes** de la red recupera la sal y
pimienta está pendiente (`docs/defensa.md`, sección 6).

Las correcciones tienen además un límite que la auditoría dejó claro: **ninguna se puede aplicar a ciegas**. CLAHE
sobre imágenes limpias baja el F1 del clásico de 0,954 a 0,323, y la inversa del brillo claro empeora el nivel 3
(0,852 → 0,795) porque lo saturado en 255 no se recupera (sección 9.4). Un sistema real necesita primero estimar qué
perturbación hay y solo después corregirla.

**Lo que sí comparten.** Los dos siguen sabiendo **que** hay sabotaje aunque se equivoquen en **cuál**: el AUC
binaria es ≥ 0,96 en 36 de las 38 combinaciones de modelo y condición; las excepciones son la CNN con sal y
pimienta en los niveles 2 y 3 (sección 8.5). Para una alarma binaria, que es la que despierta al operador, la
degradación es mucho menor que lo que sugiere el F1 macro.

Un matiz importante: la auditoría se hizo solo sobre la prueba de la ubicación 1 (sección 11.3), es decir, en el
dominio donde la CNN ya acierta. La robustez de la CNN ante perturbaciones **en una cámara nueva** no está medida, y
por lo visto en la sección 10.3 no debería suponerse.

## 10.6 Tiempo por cuadro y datos necesarios

**Tiempo.** En la auditoría, con el mismo procedimiento para los dos módulos, la mediana de las 19 condiciones es
12,4 ms para el clásico y 51,9 ms para la CNN, unas 4 veces más (`results/auditoria/tabla.csv`). Las mediciones de
las secciones 7 y 6 siguen procedimientos distintos y no se deben dividir entre sí: el clásico da una media de
12,6 ms sobre imágenes decodificadas, incluida la reducción de tamaño (`results/clasico/tiempo.json`), y la CNN una
media de 81,8 ms sobre imágenes ya reducidas a 224×224 (`results/cnn/lr1e-2_b32_frozen/evaluacion/tiempo.json`).
Los tiempos dependen de la máquina y de su carga (`docs/cnn.md`, T65). En cuadros por segundo, 12,6 ms son unos 79
(sección 7.1) y 81,8 ms unos 12. Para una cámara bastan los dos; la diferencia importa cuando un mismo equipo vigila
muchas cámaras o cuando se agregan filtros correctivos antes de clasificar.

**Datos.** El clásico ajusta 5 parámetros de decisión con el conjunto de entrenamiento (tabla 5.1) y necesita 8 referencias de cada cámara. Un
umbral se calibra con pocas imágenes: en el experimento de la sección 5.6, 60 por clase bastaron para separar
borrosas de nítidas con exactitud balanceada de 0,996. La CNN entrena 5 124 pesos con las 1930 imágenes de
entrenamiento, y eso solo es posible porque la base de 2 257 984 parámetros llega ya entrenada con ImageNet
(sección 6). Entrenarla tomó 177,9 s (`results/cnn/lr1e-2_b32_frozen/run.json`). La diferencia de fondo no es la
cantidad sino la **variedad**: el clásico necesita imágenes de la cámara donde se instala; la CNN necesitaría
imágenes saboteadas de muchas cámaras, días y escenas para no aprender la de una sola, y ese conjunto no existe en
este proyecto.

## 10.7 Límites de la comparación

- **Una sola cámara de entrenamiento.** Todo lo aprendido viene de la ubicación 1; la generalización se mide con una
  sola cámara más, sin clase tapada (sección 3.4).
- **Clase girada pequeña:** 9 imágenes de prueba en la ubicación 1.
- **Evaluaciones repetidas:** la prueba y la cámara no vista se miraron más de una vez durante el desarrollo de los
  dos módulos (sección 11.1 y `docs/cnn.md`, T63). Las cifras son optimistas para ambos.
- **Una sola configuración de CNN.** La comparación es con MobileNetV2 congelada y una cabeza lineal; no se probó
  descongelar más capas ni otra arquitectura en la cámara no vista. El resultado no permite afirmar que *ninguna*
  CNN generalice, solo que esta no lo hizo con estos datos.
- **Sin validación regional ni física.** La prueba en Medellín y la captura en Armenia no están terminadas, y la
  auditoría con luz real (T75) depende de esa captura (sección 8.6).
- **Causas de la CNN sin confirmar.** Todo lo que se dice sobre por qué falla la CNN es hipótesis hasta tener los
  mapas de Grad-CAM (T67).

## 10.8 Consecuencia

Los resultados no apuntan a elegir un módulo, sino a **combinarlos según lo que cada uno hace bien**: el clásico
generaliza a una cámara nueva con una calibración de 8 imágenes, da muy pocas falsas alarmas y su fallo se explica y
se corrige; la CNN resiste mejor la luz y el ruido leve en la cámara que conoce. La propuesta de un sistema híbrido
y lo que falta medir para sostenerla están en la sección 12.
