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
- **Pendiente:** probar con la webcam real del portátil (FPS y calibración en vivo). Esta prueba no se ha hecho.
- Las teclas de filtros están probadas por función, no en la ventana.
