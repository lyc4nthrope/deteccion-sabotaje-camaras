# Datos

Las imágenes **no se suben al repositorio** (pesan mucho y tienen su propia licencia). Solo se versiona la
división de datos en `data/splits/`.

## 1. Dataset de entrenamiento: Bremen

- **Nombre:** ComNets Camera Tampering Images Datasets (Universidad de Bremen)
- **Descarga:** https://nc.uni-bremen.de/index.php/s/GCsBcswrQf9kBXC
- **Contenido:** 1800 normales, 600 desenfocadas, 600 giradas, 600 con lente tapado
- **Licencia:** no indicada en la página; uso académico citando el artículo
- **Cita:** S. Attarha, K. Shanmugi y A. Förster, "Poster: Camera Tampering Detection for Outdoor IoT Systems",
  EWSN 2024. arXiv:2602.05706.
- **Ubicación local:** `data/raw/bremen/`

## 2. Dataset regional de prueba: Medellín

- **Nombre:** Dataset for Detecting Motorcyclists in Pedestrian Areas
- **Descarga:** https://zenodo.org/records/7935299 (`PedestrianDataSet.zip`, 643 MB)
- **Contenido:** 6324 imágenes JPG de 1280×720 de 80 cámaras CCTV públicas de Medellín (Secretaría de Movilidad).
  No tiene sabotaje: se genera por código (tarea T18).
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
