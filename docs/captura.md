# Protocolo de captura regional (T80)

Captura propia en Armenia, Quindío (D3). Sirve para la demo y su calibración, para la auditoría física (T75) y
como muestra de sabotaje real que complementa el sabotaje sintético de Medellín (D12).

Las imágenes se guardan en `data/propios/` y **no se versionan** (ver `.gitignore`).

## 1. Escena y permisos

- **Lugar:** entrada de una casa, edificio o local en Armenia, Quindío, con permiso escrito del dueño.
- **Cámara fija:** webcam o celular sobre trípode o soporte fijo. No moverla durante una sesión.
- **Encuadre:** la misma vista para las 4 clases. Anotar la altura y el ángulo del soporte en la hoja de sesión.
- **Resolución:** la misma en todas las sesiones (preferiblemente 1280×720). Anotarla.

## 2. Protección de datos (Ley 1581 de 2012)

- No capturar a personas identificables. Si alguien aparece en el encuadre, esperar a que salga o descartar la toma.
- Si una persona identificable no se puede evitar, pedir consentimiento antes de capturar.
- Nunca publicar imágenes con personas, ni en el informe ni en la presentación.
- Los datos propios no se suben al repositorio ni se comparten fuera del equipo.

## 3. Cómo simular cada sabotaje

| Clase | Cómo simularla | Qué anotar |
|---|---|---|
| normal | Escena sin intervención | Hora y condición de luz |
| desenfocada | Quitar el enfoque automático y mover el lente, o colocar un papel translúcido delante del lente | Nivel de desenfoque (leve, medio, fuerte) |
| girada | Girar el soporte de la cámara | Ángulo aproximado (con transportador o app de nivel) |
| tapada | Cubrir el lente con la mano, cinta o tela | Tipo de cobertura (total o parcial) |

Cada toma de sabotaje se hace con la misma escena normal, para que la única diferencia sea el sabotaje.

## 4. Conjunto de iluminación física

Para cada escena (normal y saboteada) se capturan las tres condiciones de luz:

| Condición | Montaje |
|---|---|
| Luz apagada | Sin luz artificial; solo la luz de la calle o de la ventana, si la hay |
| Lámpara | Una lámpara fija a un lado de la escena, siempre en la misma posición |
| Contraluz | Ventana detrás de la escena, en horario de la tarde o con cortina |

Las tres condiciones alimentan la auditoría física (T75), que compara el oscurecimiento real con el artificial
(T20, brillo).

## 5. Cantidades

- **~100 imágenes por clase** (normal, desenfocada, girada, tapada), repartidas entre las tres condiciones de luz.
- Si el tiempo se acorta, reducir a ~50 por clase (ver recorte progresivo en `docs/PLAN.md`, sección 5).
- Tomar las imágenes en ráfagas cortas con intervalos de unos segundos, para que la variación sea real.

## 6. Hoja de sesión

Por cada sesión anotar: fecha, hora, lugar, cámara (modelo y resolución), altura y ángulo del soporte, condición
de luz, clase capturada, número de imágenes y observaciones (personas en el encuadre, cambios de luz, etc.).

Nombre sugerido de archivo: `AAAAMMDD_HHMM_<clase>_<luz>_<n>.jpg`, por ejemplo `20261104_1830_tapada_contraluz_012.jpg`.

## 7. Verificación (criterios de T80)

- [ ] El protocolo cubre las 4 clases.
- [ ] El protocolo cubre el conjunto de iluminación física (3 condiciones).
- [ ] Las reglas de protección de datos están escritas y se cumplen antes de capturar.
- [ ] Cada sesión tiene su hoja de sesión.
