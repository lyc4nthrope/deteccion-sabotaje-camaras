# Datos

Las imágenes **no se suben al repositorio** (pesan mucho y tienen su propia licencia). Solo se versiona la
división de datos en `data/splits/`.

## 1. Dataset de entrenamiento: Bremen

- **Nombre:** ComNets Camera Tampering Images Datasets (Universidad de Bremen)
- **Descarga:** https://nc.uni-bremen.de/index.php/s/GCsBcswrQf9kBXC (carpeta compartida pública, 1,46 GB)
- **Contenido:** 5031 imágenes JPG de 2 cámaras:

  | Ubicación | Normal | Blur (desenfocada) | Rotated (girada) | Noimage (tapada) | Total |
  |---|---|---|---|---|---|
  | 1 (las del artículo) | 1800 | 600 | 600 | 600 | 3600 |
  | 2 | 753 | 260 | 167 | 251 | 1431 |

  La ubicación 1 se usa para train/val/test; la ubicación 2 es la prueba de cámara no vista (D6, tarea T19).
- **Licencia:** no indicada en la página; uso académico citando el artículo
- **Cita:** S. Attarha, K. Shanmugi y A. Förster, "Poster: Camera Tampering Detection for Outdoor IoT Systems",
  EWSN 2024. arXiv:2602.05706.
- **Ubicación local:** `data/raw/bremen/`

### Descarga por WebDAV

La carpeta compartida tiene `Location_1` y `Location_2`, cada una con 4 zips: Normal, Blur, Rotated y Noimage
(ubicación 1: 351/35/144/25 MB; ubicación 2: 590/167/137/10 MB). Se descargan con el WebDAV público:

```bash
curl -u "GCsBcswrQf9kBXC:" -o data/raw/bremen/zips/<archivo>.zip \
  "https://nc.uni-bremen.de/public.php/webdav/<ruta>/<archivo>.zip"
```

Organización local:

```
data/raw/bremen/
  zips/                 Los 8 zips descargados
  Location1_Normal/     Cada zip trae una carpeta superior con este nombre
  Location1_Blur/
  Location1_Rotated/
  Location1_Noimage/
  Location2_Normal/
  ...
```

## 2. Dataset regional de prueba: Medellín

- **Nombre:** Dataset for Detecting Motorcyclists in Pedestrian Areas
- **Descarga:** https://zenodo.org/records/7935299 (`PedestrianDataSet.zip`, 643 MB)
- **Contenido:** 6324 imágenes JPG de 1280×720 de 80 cámaras CCTV públicas de Medellín (Secretaría de Movilidad).
  No tiene sabotaje: se genera por código (tarea T18).
- **Inventario medido (T16):** 6324 archivos, todos de 1280×720, fechas 2022-10-03 a 2023-05-14. Según el nombre
  (`imagenN_AAAAMMDD_HHMMSS.jpg`) hay 55 grupos `imagenN`, pero 22 de ellos tienen 1 a 4 imágenes. Por eso el
  grupo por nombre **no** se trata como cámara fija sin verificar (ver T17). Detalle en `results/inventario_medellin.csv`.
- **Licencia:** CC BY 4.0
- **Cita:** N. Hernández-Díaz, E. Puertas, J. C. Martinez-Santos, G. Archbold, Y. Rios y Y. Peñaloza, "Dataset for
  Detecting Motorcyclists in Pedestrian Areas", Zenodo, 2023. DOI: 10.5281/zenodo.7935299. Artículo de datos en
  *Data in Brief*, DOI: 10.1016/j.dib.2023.109610.
- **Ubicación local:** `data/raw/medellin/`; sabotaje sintético en `data/medellin_sintetico/`
- **Cuidado:** contiene peatones. No se republican imágenes con personas identificables (Ley 1581 de 2012).

## 3. Captura propia: Armenia, Quindío

- Cámara fija (webcam o celular) en una escena real, con permiso y sin personas identificables.
- ~100 imágenes por clase y un conjunto con cambios reales de iluminación.
- **Ubicación local:** `data/propios/`, una carpeta por clase. Protocolo en `docs/captura.md` (tarea T80).
- **Cómo capturar (T81):** `python -m src.capture capturar --clase tapada --luz contraluz --n 50` toma una ráfaga con la webcam, nombra
  los archivos como pide el protocolo y anota la hoja de sesión (`data/propios/hoja_sesion.csv`). Después,
  `python -m src.capture anonimizar` difumina a las personas en `data/propios_anonimas/` (requiere `requirements-anonymize.txt`).
