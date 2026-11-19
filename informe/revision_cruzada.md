# Revisión cruzada del informe (T103)

Cada uno lee las secciones del otro como lo haría la profesora: busca cifras que no coinciden con `results/`,
afirmaciones sin evidencia y partes que sigan a medias. Se anota cada hallazgo con la sección, la línea y la
corrección propuesta. Una cifra se verifica contra el CSV que la cita, no contra otra sección del informe.

## Reparto

| Revisa | Secciones |
|---|---|
| **Santiago** lee lo de Cristhian | 1, 2, 3.1–3.4, 4, 5, 7.1, 7.3, 8, 11 y A |
| **Cristhian** lee lo de Santiago | 3.5, 3.6, 6, 7.2, 9.5 y `docs/defensa.md` sección 6 |
| **Los dos** | 9.1–9.4, 10 y 12 (las escribieron entre los dos) |

## Lista de comprobación

**Cifras**
- [ ] Cada número tiene su archivo de `results/` y el archivo existe en `main`.
- [ ] Las mismas cifras coinciden entre secciones: F1 0,954 y 0,967 (prueba), 0,920 y 0,580 (cámara no vista),
      falsas alarmas 0,5 % y 0,0 %, tiempos 12,6 y 81,8 ms, AUC de la CNN en prueba 1,000 (0,99996).
- [ ] "Mejor F1" usa la regla de la sección 8 (diferencia mayor que 0,02): 10 condiciones de la CNN, 4 del clásico.
- [ ] La cifra de la cámara no vista dice que se miró más de una vez (sección 11).

**Honestidad**
- [ ] Lo que no se midió está dicho como **hipótesis** o **no medido**: causas de la CNN con JPEG y ruido gaussiano,
      sistema híbrido, tapa con la mano, degradación física.
- [ ] Nada afirma que la CNN "mire el sabotaje" o "el contexto" (Grad-CAM no lo prueba).
- [ ] La captura propia y Medellín figuran como **no evaluadas**, de forma consistente en las secciones 3, 11 y 12.
- [ ] El test se mencionó como mirado varias veces donde se eligió un filtro o un parámetro con él.

**Coherencia**
- [ ] No queda ningún marcador "Redacta …" ni "Pendiente" en el cuerpo de las secciones.
- [ ] Las referencias `[n]` existen en `informe/referencias.md` y están numeradas sin saltos.
- [ ] Las tablas y figuras están numeradas y citadas en el texto; cada ruta de imagen existe.
- [ ] Los nombres de archivo y comandos coinciden con el repositorio (`src/…`, `results/…`).

**Defensa**
- [ ] Para cada ataque de `docs/defensa.md` el informe tiene el mismo diagnóstico (síntoma, causa, filtro, mejora).
- [ ] La frase del sistema híbrido dice lo mismo en la sección 12.2, en `docs/defensa.md` y en la presentación.

## Cosas que ya se sabe que hay que mirar

- **Sección 3.5 (Medellín) y 3.6 (Armenia):** el código de Medellín vive en los PR #2 y #31, todavía sin fusionar a
  `main`. Si no se fusionan, la sección debe seguir diciendo "preparado, no evaluado". Si se toman las fotos de
  Armenia, hay que cambiar el estado de la 3.6 y de la 11.
- **Sección 8.6 (Pendiente):** sigue con la lista de T75, T76 y T77. Confirmar qué se declara como limitación en la
  sección 11.
- **Sección 11:** una de sus frases ("se prueba con la captura propia") puede leerse como que la captura se hizo.
- **ORB, SIFT y KAZE (T77):** no tiene tarea cumplida; la tabla 5.3 de la sección 5 solo compara su separación.

## Registro de hallazgos

| Sección | Línea | Hallazgo | Corrección propuesta | Quién corrige | Hecho |
|---|---|---|---|---|---|
| | | | | | |
