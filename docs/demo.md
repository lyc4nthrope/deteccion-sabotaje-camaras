# Demo en vivo (T07, T82, T84, T85)

La demo corre en el portátil con la webcam (D11). No usa Colab.

## Instalación (T07)

```bash
python -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt
```

Usa Python 3.11 o 3.12: TensorFlow no tiene wheels para Python 3.14. OpenCV debe ser menor que 5 (`opencv-python<5`): OpenCV 5 quitó KAZE, que usa T38.

## Ejecutar

```bash
python -m src.demo --source 0                       # webcam
python -m src.demo --source 0 --cnn modelo_demo_lr1e-2   # webcam + CNN (T84)
```

El modelo de la CNN es `results/cnn/<carpeta>/model.keras`. No se versiona (ver `.gitignore`); hay que entrenarlo con `python -m src.train_cnn`.

## Calibración (T82)

Al empezar, la demo toma 8 cuadros de la escena en vivo como referencias normales para ORB. Por eso la cámara debe estar en su posición final y sin nadie delante. Si se mueve, hay que reiniciar la demo.

Con una webcam, antes de esos 8 cuadros descarta 30 (`WARMUP_FRAMES`, unos 1,5 s): la webcam arranca con una exposición por defecto y la corrige en sus primeros cuadros (medido: gris medio de 74 a 103 en unos 2 s). Con referencias tomadas en ese intervalo, la escena quieta daba 28–33 coincidencias y giros falsos de ~1°, por encima del límite de 0,94°, y salía "girada". Un archivo de video no descarta nada.

No toques el portátil durante la demo: mover la pantalla 1° o 2° ya supera el límite de ángulo y da "girada".

## Teclas (T85)

| Tecla | Efecto |
|---|---|
| `c` | Activa o desactiva CLAHE |
| `g` | Activa o desactiva la gamma inversa (nivel 2) |
| `m` | Activa o desactiva la mediana |
| `q` | Sale |

Con filtros activos, la pantalla muestra la predicción antes y después del filtro, y los FPS.

## Pruebas sin cámara

```bash
python -m src.demo --source video.mp4 --headless --max-frames 100
```

## Estado

- Probado con un video de Bremen: 52 cuadros tras la calibración, 32 normales y 20 desenfocadas, todas bien clasificadas por el clásico. Sin ventana y sin webcam.
- Probado con la webcam de un portátil (Linux, Python 3.13, CPU), con el clásico y la CNN a la vez, el 2026-11-18:
  - Escena quieta, después del calentamiento: 230 de 230 cuadros normales en el clásico, 594 a 751 coincidencias ORB, ángulo máximo 0,08°.
  - Lente tapado con la mano: 105 de 110 cuadros "tapada"; los otros 5 son la transición.
  - Hoja de papel a 1–2 cm del lente: "desenfocada" (Laplaciano 8–10, dispersión del matiz 25–49). Pegada al lente: "tapada" (dispersión del matiz 4–5, menor de 5,4).
  - Luz apagada: el Laplaciano cae a 4–10 y el clásico dice "desenfocada". La gamma inversa (`g`) sube el gris medio de 20 a 71, como predice 255 · (20/255)^0,5 = 71,4, pero no devuelve los bordes: sigue "desenfocada".
  - 10 a 16 FPS con los dos modelos; unos 7,5 FPS a oscuras, porque la webcam alarga la exposición.
  - La CNN se equivoca a menudo en esta cámara que no vio: dijo "girada" en los 230 cuadros de la escena quieta (confianza de 0,54 a 0,94).
- La tecla `g` se probó en la ventana; `c` y `m` solo por función.
