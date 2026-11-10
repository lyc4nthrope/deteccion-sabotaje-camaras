# Contexto regional: cámaras de vigilancia fuera de servicio y vandalizadas (T06)

Fuentes verificadas el **4 de noviembre de 2026**: las cinco páginas estaban disponibles y las citas son textuales.

## 1. El problema en Colombia

Las cámaras de vigilancia son una herramienta de seguridad que las autoridades usan para capturar y judicializar
delincuentes, pero una parte importante queda fuera de servicio por fallas, desgaste, robo de cableado y
vandalismo.

| Ciudad | Hecho | Fuente |
|---|---|---|
| **Bucaramanga** (2025) | "De las 808 cámaras de seguridad instaladas en Bucaramanga, 526 están actualmente en funcionamiento, mientras que 208 permanecen fuera de servicio debido a fallas técnicas, desgaste o vandalizadas." | Blu Radio, 4 de noviembre de 2025 [1] |
| **Bucaramanga** (2025) | Robo de "más de 2.000 metros de cableado del sistema de videovigilancia"; "delincuentes han cortado cables, lanzado piedras, incluso disparado a las cámaras". En ese momento, cerca del 53 % de las 808 cámaras funcionaba de forma óptima. | Blu Radio, 15 de abril de 2025 [2] |
| **Bogotá** (2022) | La Contraloría Distrital "encontró que había 719 cámaras de seguridad fuera de servicio"; causas: "la falta de energía, a causa del robo del cableado", "el vandalismo" y obras de entidades como el IDU y Enel-Colombia. | El Tiempo, 28 de septiembre de 2022 [3] |

Notas de lectura:

- En la fuente [1], 526 en funcionamiento más 208 fuera de servicio suman 734, no 808; la nota no explica las 74
  restantes. Se cita textual.
- La fuente [3] es de 2022: se usa como antecedente, no como dato actual.

## 2. Relevancia en Armenia (Quindío)

| Hecho | Fuente |
|---|---|
| En un robo de 30 millones de pesos en Armenia (abril de 2026), "las cámaras de vigilancia y la denuncia ciudadana fueron herramientas decisivas" para la captura. El centro de monitoreo "integra las cámaras de seguridad del gobierno departamental". | Pulzo, 22 de abril de 2026, con base en La Crónica del Quindío [4] |
| La Alcaldía de Armenia denuncia vandalismo contra la red de semáforos: "hurto de cableado de cobre y tarjetas controladoras", "cerca 12 controladores" vandalizados en el año y "más de seis —6— kilómetros de cable hurtado". | Alcaldía de Armenia, 20 de mayo de 2026 [5] |

Notas de lectura:

- La fuente [5] **no habla de cámaras**, sino de semáforos. Se usa como evidencia de que la infraestructura
  electrónica urbana de Armenia sufre vandalismo y robo de cableado, el mismo tipo de daño que afecta a las cámaras
  en las otras ciudades.
- La fuente [4] es una nota de Pulzo que resume una publicación de La Crónica del Quindío; para el informe conviene
  citar también la nota original de La Crónica si se consigue.

## 3. Por qué este proyecto

Si una cámara se tapa, se desenfoca o se gira, deja de servir para el monitoreo y para las capturas, y el sabotaje
puede pasar inadvertido hasta que se necesitan las grabaciones. Un sistema que detecte automáticamente estos
sabotajes permite reaccionar a tiempo. En Bucaramanga, por ejemplo, una de cada cuatro cámaras estaba fuera de
servicio en 2025 [1].

## 4. Datos usados y nota aprobada por la profesora

- **No se encontró un dataset público regional de cámaras saboteadas.** Se buscó en datos.gov.co y en
  repositorios académicos, y tampoco hay cámaras web públicas de Armenia o Salento que puedan usarse.
- Por eso, **el método se desarrolló y entrenó con el dataset de la Universidad de Bremen** (Alemania): se usó para
  entender el problema y construir el sistema. **El método no depende de esa ciudad: se aplica igual a un dataset
  regional.** Esta nota fue aprobada por la profesora.
- **Dato regional usado:** el dataset de cámaras CCTV públicas de Medellín (Hernández-Díaz et al., Zenodo, 2023,
  DOI 10.5281/zenodo.7935299, licencia CC BY 4.0), con sabotaje sintético (T16–T18), y la captura propia en Armenia
  (T80–T81).

## Fuentes

1. Redacción BLU Radio Santander. "Videovigilancia en Bucaramanga: de 808 cámaras sirven 526". Blu Radio, 4 de
   noviembre de 2025. https://www.bluradio.com/regiones/santanderes/videovigilancia-en-bucaramanga-de-808-camaras-sirven-526-rg10
2. V. Rincón. "Robaron 2.000 metros de cable de cámaras de seguridad en Bucaramanga". Blu Radio, 15 de abril de 2025.
   https://www.bluradio.com/regiones/santanderes/robaron-2-000-metros-de-cable-de-camaras-de-seguridad-en-bucaramanga-rg10
3. G. Reinoso. "700 cámaras de seguridad fuera de servicio por robo, obras y falta de energía". El Tiempo, 28 de
   septiembre de 2022. https://www.eltiempo.com/bogota/camaras-de-seguridad-fuera-de-servicio-por-robo-vandalismo-y-obras-706006
4. "Robo millonario en Armenia: así fue la persecución y el uso de tecnología para capturar a los asaltantes".
   Pulzo (con base en La Crónica del Quindío), 22 de abril de 2026.
   https://www.pulzo.com/nacion/robo-de-30-millones-en-armenia-camaras-mapas-de-calor-y-reaccion-policial-permiten-captura-PP5150773A
5. Alcaldía de Armenia, Secretaría de Tránsito y Transporte. "Alcaldía de Armenia rechaza actos vandálicos contra
   red semafórica y llama a la ciudadanía a proteger la infraestructura pública". 20 de mayo de 2026.
   https://armenia.gov.co/alcaldia-de-armenia-rechaza-actos-vandalicos-contra-red-semaforica-y-llama-a-la-ciudadania-a-proteger-la-infraestructura-publica
