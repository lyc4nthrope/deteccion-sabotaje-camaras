# Ensayo de la defensa con la demo (T94)

Uno ataca la cámara en vivo y el otro **diagnostica con números y activa el filtro**; después se cambian los
papeles. El objetivo es que los dos puedan explicar cualquier ataque sin leer. La demo está en `docs/demo.md`; las
causas y los filtros, en `docs/defensa.md`.

## Antes de empezar (10 minutos)

- [ ] Portátil de la defensa, con `.venv312` y `pip install -r requirements.txt` (Python 3.11 o 3.12, OpenCV < 5).
- [ ] La CNN entrenada: `results/cnn/modelo_demo_lr1e-2/model.keras` (no se versiona; se entrena con `python -m src.train_cnn`).
- [ ] La webcam abre: `python -m src.demo --source 0`.
- [ ] Cámara en su **posición final y sin nadie delante**: la calibración toma 8 cuadros de la escena en vivo y,
      antes, descarta 30 (unos 1,5 s) mientras la webcam ajusta la exposición.
- [ ] Luz estable, sin tocar la pantalla ni el soporte durante la demo: moverla 1° o 2° supera el límite de ángulo
      (0,94°) y da "girada".
- [ ] Una hoja de papel, una linterna o lámpara, y la mano para tapar.

## Ronda de ataques

El que ataca elige el ataque sin avisar. El que defiende dice, **antes de mirar la pantalla**, qué predice y por qué,
y después confirma con los números de la pantalla (valor / umbral).

| # | Ataque en vivo | Qué debe decir el clásico | Diagnóstico que hay que dar | Filtro |
|---|---|---|---|---|
| 1 | Escena quieta | normal | Coincidencias ORB muy sobre 12,5 y ángulo bajo 0,94° (medido: 594 a 751 y 0,08° como máximo) | — |
| 2 | Tapar el lente con la mano | tapada | Sin coincidencias, Laplaciano bajo 18,9 y dispersión del matiz bajo 5,4°: una sola tonalidad | — |
| 3 | Hoja de papel a 1 o 2 cm del lente | desenfocada | Laplaciano de 8 a 10 (bajo 18,9) y dispersión del matiz de 25 a 49, **sobre** 5,4: hay color, no es tapa | — |
| 4 | Hoja de papel pegada al lente | tapada | Dispersión del matiz de 4 a 5, bajo 5,4 | — |
| 5 | Apagar la luz | desenfocada | El Laplaciano escala con g² y cae a 4 a 10; el gris medio cae a unos 20 | `g` sube el gris medio de 20 a 71 (255 · (20/255)^0,5), **pero no devuelve los bordes**: sigue "desenfocada" |
| 6 | Girar la cámara a otra parte de la escena | girada | Pocas coincidencias con las referencias de esta cámara, o ángulo mayor a 0,94° | — |
| 7 | Luz de linterna sobre el lente | normal o girada | La saturación en 255 borra textura y bajan las coincidencias | La inversa del brillo no recupera lo cortado en 255 |

Si el resultado no coincide con esta tabla, **no es un fallo del ensayo**: es lo que hay que explicar. La demo ya
mostró dos casos así (filas 5 y 7).

## El filtro en vivo

1. El atacante provoca el fallo.
2. El defensor lo diagnostica en voz alta con el descriptor que falla y su umbral.
3. Activa la tecla (`m` mediana, `g` gamma inversa, `c` CLAHE) y se lee la predicción **antes y después** del filtro.
4. Dice qué pasó, aunque el filtro no ayude. Un filtro que no ayuda es un resultado, y tiene explicación.

**No activar CLAHE a ciegas**: sin ataque baja el F1 del clásico de 0,954 a 0,323 y manda las borrosas y tapadas a
"girada" (`docs/defensa.md`, sección 4). La mediana es la única segura de dejar activa.

## Riesgos conocidos de la demo

- **La CNN se equivoca en esta cámara.** Con la escena quieta dijo "girada" en los 230 cuadros (confianza de 0,54 a
  0,94). No es un fallo del código: la CNN nunca vio esa cámara (F1 0,580 en la cámara no vista). Decirlo **antes**
  de que lo pregunten, y usarlo como la prueba en vivo de la sección 7 del informe.
- **Los FPS bajan con la luz apagada** (unos 7,5), porque la webcam alarga la exposición.
- **Recalibrar** si alguien mueve la cámara: hay que reiniciar la demo.
- **La tecla `g` se probó en la ventana; `c` y `m` solo por función.** Probarlas en vivo en el ensayo.
- **Tapar con la mano**: la lectura mide el color de la piel; si la mano deja pasar luz con textura puede salir
  "desenfocada" (`docs/defensa.md`, sección 5).

## Registro del ensayo

Completar durante el ensayo: la fecha, quién ataca y quién defiende, y para cada fila la predicción real.

| Fecha | Ataque (fila) | Ataca | Defiende | Predicción real | ¿Coincide? | Explicó sin leer |
|---|---|---|---|---|---|---|
| | | | | | | |

Se repite con los papeles cambiados. El ensayo está bien cuando los dos explican **todas** las filas sin leer, y
saben decir qué no está medido (ruido digital en vivo, JPEG y ruido gaussiano fuerte en la CNN).
