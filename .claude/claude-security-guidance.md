# Security guidance: deteccion-sabotaje-camaras

Academic computer-vision project. Python code reads downloaded image datasets and writes CSV/figures.

## Must never happen

- Committing data: images under `data/raw/`, `data/propios/` or `data/medellin_sintetico/`, trained models
  (`*.h5`, `*.keras`, `*.pt`, `*.onnx`) or webcam captures. Only `data/README.md` and `data/splits/*.csv` are
  versioned.
- Publishing images with identifiable people (Medellín CCTV dataset, own captures): Colombian Ley 1581 de 2012.
  Figures for the report must use frames without people or blur faces.
- Extracting archives without checking member paths (absolute paths or `..` components): check before
  extracting any downloaded zip.
- Executing code from downloaded datasets, or loading pickled objects (`pickle`, `np.load(allow_pickle=True)`,
  untrusted model files) from untrusted sources.
- Hard-coding credentials or tokens. The Bremen WebDAV share token is public and documented in `data/README.md`;
  no other secret belongs in the repository.

## Use instead

- Paths: build them from `src.data.ROOT` / `BREMEN_ROOT`; read images with `cv2.imread` and check for `None`.
- Data loading: `src.data.load_split()` for the split; never re-split ad hoc in notebooks.
- Webcam demo: local only (`cv2.VideoCapture`); do not save frames unless the capture protocol (docs/captura.md)
  allows it.
