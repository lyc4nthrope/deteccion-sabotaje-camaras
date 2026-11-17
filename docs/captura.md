# Protocolo de captura regional (T80)

Captura propia en Armenia, Quindío (D3). Sirve para la demo y su calibración, para la auditoría física (T75) y
como muestra de sabotaje real que complementa el sabotaje sintético de Medellín (D12).

Las imágenes se guardan en `data/propios/` y **no se versionan** (ver `.gitignore`). Para tomar las ráfagas, nombrar
los archivos y llenar la hoja de sesión se puede usar `python -m src.capture capturar` (`src/capture.py`, PR #32).

## 1. Escena y permisos

- **Lugar:** entrada de una casa, edificio o local en Armenia, Quindío, con permiso escrito del dueño.
- **Cámara fija:** webcam o celular sobre trípode o soporte fijo. No moverla durante una sesión.
- **Encuadre:** la misma vista para las 4 clases. Anotar la altura y el ángulo del soporte en la hoja de sesión.
- **Marcar la posición:** poner cinta en el suelo o en la mesa alrededor del soporte antes de empezar. Después de
  girar la cámara (clase girada) o de tapar y destapar el lente, hay que volver exactamente al mismo encuadre.
- **Resolución:** la misma en todas las sesiones (preferiblemente 1280×720). Anotarla.

## 2. Protección de datos (Ley 1581 de 2012)

- **No capturar a personas identificables.** Si alguien aparece en el encuadre, esperar a que salga y descartar la toma.
- **Consentimiento escrito, previo e informado,** si una persona identificable no se puede evitar (por ejemplo, vive o
  trabaja en el lugar). Se pide **antes** de capturar con la hoja de autorización del anexo; sin esa hoja firmada no
  se captura con esa persona en el encuadre.
- **Difuminar antes de usar o compartir:** `python -m src.capture anonimizar` o `python -m src.anonymize <origen>
  <destino>` (YOLOv8n, ver `requirements-anonymize.txt`) escriben una copia con las personas difuminadas y no tocan
  la carpeta de origen. Revisar a mano la copia difuminada: el detector puede fallar con personas pequeñas, de
  espaldas o parcialmente tapadas. Si la copia no cubre a alguien, se descarta la toma.
- **Imágenes crudas con personas:** una vez comprobada la copia difuminada se borran las crudas que contengan
  personas (el difuminado no sustituye al consentimiento mientras exista la imagen original).
- Nunca publicar imágenes con personas, ni en el informe ni en la presentación.
- Los datos propios no se suben al repositorio ni se comparten fuera del equipo.

## 3. Cómo simular cada sabotaje

| Clase | Cómo simularla | Qué anotar |
|---|---|---|
| normal | Escena sin intervención | Hora y condición de luz |
| desenfocada | Quitar el enfoque automático y mover el lente, o colocar un papel translúcido delante del lente | Nivel de desenfoque (leve, medio, fuerte) |
| girada | Paneo: girar la cámara en horizontal hacia otra parte de la escena, como en Bremen. No inclinar la imagen | Lado del paneo y ángulo aproximado (con transportador o app de nivel) |
| tapada | Cubrir el lente con la mano, cinta o tela. **La mitad total y la mitad parcial** | Tipo de cobertura y si es total o parcial |

Cada toma de sabotaje se hace con la misma escena normal, para que la única diferencia sea el sabotaje.

## 4. Conjunto de iluminación física

Para cada escena (normal y saboteada) se capturan las tres condiciones de luz:

| Condición | Montaje |
|---|---|
| Luz apagada | Sin luz artificial; solo la luz de la calle o de la ventana, si la hay. En interiores, cortina cerrada y luces apagadas |
| Lámpara | Una lámpara fija a un lado de la escena, siempre en la misma posición |
| Contraluz | Ventana detrás de la escena, en horario de la tarde o con cortina |

Las tres condiciones alimentan la auditoría física (T75), que compara el oscurecimiento real con el artificial
(T20, brillo).

## 5. Orden de la sesión: la luz fija por bloques

En exteriores "luz apagada" exige noche y "contraluz" exige tarde, y las dos no caben en una tarde. Por eso la
**luz se fija por bloques** y dentro de cada bloque se toman las 4 clases, sin cambiar la luz:

1. Bloque de **luz apagada** (de noche, o con la cortina cerrada en interiores).
2. Bloque de **lámpara**.
3. Bloque de **contraluz** (la ventana detrás de la escena, de tarde).

Dentro de cada bloque: primero las **referencias**, después las 4 clases con la cámara en la posición marcada
(sección 1), y la normal antes que los sabotajes. En interiores un bloque lleva unos 10 a 20 minutos y toda la
sesión 1 a 2 horas; en exteriores, cada bloque en su horario.

## 6. Cantidades

- **~100 imágenes por clase** (normal, desenfocada, girada, tapada), repartidas entre las tres condiciones de luz:
  unas **33 por combinación de clase y luz**. De las 100 tapadas, unas 50 **totales** y 50 **parciales**.
- **Referencias, además de las 100:** unas **8 normales por condición de luz** (24 en total), tomadas al principio de
  cada bloque. El método clásico compara cada imagen contra referencias de **la misma cámara** (T33) y la
  calibración de una cámara nueva toma 8 (`calibration_split`); estas **no entran en la evaluación**.
- Si el tiempo se acorta, reducir a ~50 por clase (ver recorte progresivo en `docs/PLAN.md`, sección 5).
- Tomar las imágenes en ráfagas cortas con intervalos de unos segundos, para que la variación sea real.

## 7. Cómo evaluar sin fugas de datos

Las fotos de una misma ráfaga, tomadas con segundos de diferencia, son casi duplicados: si una queda en
calibración y otra en prueba, la prueba mide memoria y no generalización. Por eso, al armar la evaluación (T83):

- **Una ráfaga entera va a un solo conjunto.** Se identifica por clase, luz y la fila de la hoja de sesión.
- Las ráfagas de **referencias** nunca se usan como prueba.
- Si una luz o un sabotaje se repite en otra sesión, se trata como ráfaga distinta.

## 8. Hoja de sesión

Por cada sesión anotar: fecha, hora, lugar, cámara (modelo y resolución), altura y ángulo del soporte, condición
de luz, clase capturada, número de imágenes y observaciones (personas en el encuadre, cambios de luz, si la ráfaga
es de referencias, etc.). `src/capture.py` la llena en `data/propios/hoja_sesion.csv`.

Nombre sugerido de archivo: `AAAAMMDD_HHMM_<clase>_<luz>_<n>.jpg`, por ejemplo `20261104_1830_tapada_contraluz_012.jpg`.

## 9. Verificación (criterios de T80)

- [ ] El protocolo cubre las 4 clases.
- [ ] El protocolo cubre el conjunto de iluminación física (3 condiciones) y el orden de la sesión por bloques de luz.
- [ ] Las reglas de protección de datos están escritas y se cumplen antes de capturar: sin personas, o con
  consentimiento escrito, y con copia difuminada revisada a mano.
- [ ] Hay un juego de referencias normales separado (unas 8 por luz) que no se usa como prueba.
- [ ] La evaluación separa por ráfaga o sesión.
- [ ] Cada sesión tiene su hoja de sesión.

## Anexo. Hoja de autorización (Ley 1581 de 2012)

> **Autorización para el tratamiento de imágenes con fines académicos**
>
> Yo, ______________________ (nombre completo), identificado con documento n.º ______________, autorizo de forma
> previa, expresa e informada a ______________________ y ______________________ (estudiantes del curso de Visión
> Artificial) para captar mi imagen en ______________________ (lugar) el ____ / ____ / ________.
>
> - **Finalidad:** construir y evaluar un sistema que detecta cuando una cámara de vigilancia es tapada, desenfocada o
>   girada. Las imágenes se usan solo dentro del equipo y del trabajo académico.
> - **Tratamiento:** se difumina mi imagen antes de usarla o compartirla; las imágenes originales donde yo aparezca se
>   borran una vez revisada la copia difuminada. No se publicarán.
> - **Mis derechos:** conocer, actualizar y rectificar mis datos; solicitar prueba de esta autorización; ser informado
>   del uso que se les dio; revocar la autorización y pedir la supresión de mis imágenes; y presentar quejas ante la
>   Superintendencia de Industria y Comercio. Puedo ejercerlos escribiendo a ______________________.
>
> Firma: ______________________   Fecha: ____ / ____ / ________
