# 4. Marco teórico

Esta sección resume las herramientas que usan los dos módulos y por qué cada una sirve para un tipo de sabotaje.
Los detalles de implementación están en la sección 5 (módulo clásico) y en la sección 6 (CNN).

## 4.1 Nitidez: varianza del Laplaciano

El Laplaciano es la suma de las segundas derivadas de la intensidad, ∇²I = ∂²I/∂x² + ∂²I/∂y². En una imagen
digital se aproxima con un núcleo de 3×3 (sección 9.1) y responde con valores grandes en los bordes y cercanos a
cero en las zonas lisas. La **varianza de esa respuesta** sobre toda la imagen es una medida clásica de enfoque
[8]: una imagen nítida tiene muchos bordes y una varianza alta.

**Por qué sirve.** Un desenfoque actúa como un filtro pasa bajos: elimina las frecuencias altas, que son las que
forman los bordes, y la varianza del Laplaciano cae. Una tapa también la hace caer, porque deja una superficie casi
sin textura. Su punto débil se deduce de la misma definición: como es una segunda derivada, amplifica el ruido, que
también es de frecuencia alta (sección 9.1).

## 4.2 Histograma y entropía

El histograma cuenta cuántos píxeles tiene cada nivel de gris. Normalizado, pᵢ es la proporción de píxeles con el
nivel i, y la entropía de Shannon

  H = −Σ pᵢ log₂ pᵢ

mide, en bits, cuánta información tiene la imagen: vale 0 si todos los píxeles tienen el mismo nivel y como máximo
8 bits con 256 niveles igual de ocupados. La desviación estándar de los niveles de gris mide el contraste global.

**Por qué sirve.** Una cámara tapada produce una imagen casi uniforme: pocos niveles ocupados, entropía y contraste
bajos. En los datos reales, sin embargo, una escena nocturna también es oscura y poco contrastada, y por eso el
calibrador terminó eligiendo un descriptor de color para la tapa (sección 5.4).

## 4.3 Espacio de color HSV

HSV separa el color en matiz (H, el tono, un ángulo), saturación (S, la pureza del color) y valor (V, el brillo).
A diferencia de RGB, el brillo queda aislado en un solo canal, así que un cambio de iluminación altera sobre todo V
y deja el matiz casi igual. En OpenCV H va de 0 a 179 (grados divididos por 2) y S y V de 0 a 255.

Como el matiz es un ángulo, su dispersión se mide con estadística circular: cada matiz se representa como un
vector unitario, R es la longitud del vector medio y la desviación circular es √(−2 ln R). Solo se usan los
píxeles con color (S ≥ 40 y V ≥ 40), porque un píxel gris o muy oscuro no tiene un matiz definido
(`src/features.py`).

**Por qué sirve.** Una tapa suele ser de un solo color, o de ningún color si está a oscuras: la dispersión del
matiz es casi 0. Una escena real, incluso de noche, conserva tonos distintos (luces, señales, vegetación). El
mismo espacio permite umbralizar tonos de piel para una mano que tapa la cámara (sección 5.2).

## 4.4 Puntos característicos, emparejamiento y RANSAC

Un detector de puntos característicos encuentra lugares de la imagen que se pueden volver a localizar en otra
vista (esquinas, manchas) y un descriptor resume la vecindad de cada uno en un vector. Se comparan tres:

- **ORB** [9]: detecta esquinas con FAST y las describe con BRIEF orientado, un descriptor **binario** que se
  compara con la distancia de Hamming. Es el más rápido.
- **SIFT** [10]: busca extremos en una pirámide de diferencias de gaussianas y describe cada punto con
  histogramas de gradientes (128 valores reales, distancia L2). Es invariante a escala y rotación.
- **KAZE** [11]: construye el espacio de escalas con difusión **no lineal**, que suaviza las zonas planas sin
  borrar los bordes; descriptor real con distancia L2.

Para emparejar, cada punto de la imagen se compara con los de una referencia y se acepta su mejor pareja solo si
es claramente mejor que la segunda (razón de Lowe [10]: distancia₁ < r · distancia₂, con r = 0,75 en ORB y 0,7
en SIFT y KAZE, `src/features.py`).

Las parejas aceptadas todavía incluyen errores. **RANSAC** [12] estima una transformación geométrica robusta:
toma al azar el mínimo de parejas que la define, cuenta cuántas otras parejas la cumplen dentro de una tolerancia
(5 píxeles en el código) y conserva el modelo con más puntos consistentes. La transformación más general entre dos
vistas de una escena plana es una **homografía** (matriz de 3×3, 8 grados de libertad, mínimo 4 parejas). El
módulo clásico usa un caso restringido, la **transformación de similaridad** (rotación, escala uniforme y
traslación; 4 grados de libertad, mínimo 2 parejas), estimada con `cv2.estimateAffinePartial2D`, porque solo
necesita el ángulo de giro: θ = atan2(a₂₁, a₁₁) a partir de la matriz estimada.

**Por qué sirve.** Una cámara girada o apuntando a otra escena pierde coincidencias con sus propias imágenes de
referencia; si el giro es parcial y la escena todavía se superpone, RANSAC mide el ángulo. Es una comparación
**relativa a la misma cámara**, que no depende de cómo se vea la escena en términos absolutos (sección 7.3).

## 4.5 Redes neuronales convolucionales

Una CNN aprende sus propios filtros. Cada capa convolucional desliza f × f pesos sobre la entrada y produce un
mapa de activación; las primeras capas responden a bordes y texturas y las profundas a partes y objetos. El tamaño
de salida de una convolución es

  salida = ⌊(n + 2p − f) / s⌋ + 1

con n el tamaño de entrada, p el relleno por lado, f el tamaño del núcleo y s el paso (stride). Con relleno
`same` la salida es ⌈n / s⌉, así que cada capa con s = 2 divide el tamaño por 2. Al final, un pooling global
promedia cada mapa a un número y una capa densa con softmax da la probabilidad de cada clase; la red se entrena
minimizando la entropía cruzada entre esas probabilidades y la etiqueta real.

## 4.6 Transfer learning con MobileNetV2

Entrenar una CNN desde cero necesita muchas imágenes. El **transfer learning** reutiliza una red ya entrenada en
un conjunto grande (ImageNet [15]) como extractor de características: se congela la base y solo se entrena una
cabeza nueva para las clases del problema. Con 1930 imágenes de entrenamiento, de ellas 84 giradas (sección 3.3),
es la opción razonable.

MobileNetV2 [13] es una red diseñada para equipos con poca capacidad de cómputo. Usa dos ideas:

- **Convolución separable en profundidad** [16]: una convolución estándar de f × f con c canales de entrada y c′
  de salida tiene f² · c · c′ parámetros. La separable la divide en una convolución *depthwise* (un filtro f × f
  por canal, f² · c) y una *pointwise* de 1×1 que mezcla los canales (c · c′). Con 144 canales y f = 3, pasa de
  186 624 a 22 032 parámetros, unas 8,5 veces menos (`docs/cnn.md`).
- **Bloque residual invertido**: expansión 1×1 que aumenta los canales, depthwise 3×3 y proyección 1×1
  **lineal** (sin ReLU, para no destruir información en un espacio estrecho). El atajo y = x + F(x) une las capas
  estrechas, al revés que en ResNet, y facilita que el gradiente llegue a las primeras capas.

La configuración usada (entrada 224×224, base congelada de 2 257 984 parámetros y cabeza de 5 124 entrenables) se
detalla en la sección 6.

**Por qué sirve.** Las características aprendidas en ImageNet ya distinguen bordes, texturas y estructura de la
escena, que es lo que cambia con cada sabotaje: el desenfoque quita textura, la tapa quita la escena y el giro la
desplaza. Su riesgo es el opuesto al de las reglas: la cabeza aprende cualquier diferencia que separe las clases en
los datos de entrenamiento, también las que no son sabotaje (secciones 10.3 y 10.4).

## 4.7 Métricas

Para cada clase, con VP, FP y FN los verdaderos positivos, falsos positivos y falsos negativos:

  precisión = VP / (VP + FP),  recall = VP / (VP + FN),  F1 = 2 · precisión · recall / (precisión + recall)

- **F1 macro:** promedio sin pesos del F1 de cada clase presente (`src/metrics.py`). Pesa igual la clase girada,
  con 9 imágenes de prueba, que la normal, con 182; la exactitud, en cambio, la dominarían las normales.
- **Tasa de falsas alarmas:** proporción de imágenes normales clasificadas como cualquier sabotaje,
  FP / (FP + VN) en la versión binaria. Es la métrica que importa en un entorno estable: cada falsa alarma le
  cuesta tiempo a un operador.
- **Curva ROC y AUC:** para el problema binario (normal frente a saboteada), la curva ROC grafica la tasa de
  verdaderos positivos frente a la de falsos positivos al mover el umbral sobre un puntaje continuo [14]. El AUC
  es el área bajo la curva: la probabilidad de que una imagen saboteada elegida al azar tenga mayor puntaje que
  una normal elegida al azar. No depende de un umbral, así que mide si el modelo sabe **que** hay sabotaje,
  mientras que el F1 macro mide además si acierta **cuál**. El puntaje del clásico se define en la sección 5.5;
  el de la CNN es 1 − P(normal) (`src/evaluate_cnn.py`).
