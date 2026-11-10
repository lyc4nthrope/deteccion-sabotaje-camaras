# 1. Introducción y problema regional

Las cámaras de vigilancia son una herramienta de seguridad en las ciudades colombianas: las autoridades las usan
para monitorear el espacio público y para capturar y judicializar delincuentes. En Armenia (Quindío), el centro de
monitoreo integra las cámaras de seguridad del gobierno departamental, y en abril de 2026 las cámaras fueron
"herramientas decisivas" para capturar a los responsables de un robo de 30 millones de pesos [4].

Una parte importante de estas cámaras deja de funcionar. En Bucaramanga, de 808 cámaras instaladas, 208 estaban
fuera de servicio en noviembre de 2025 por "fallas técnicas, desgaste o vandalizadas" [1], es decir, una de cada
cuatro. En la misma ciudad se robaron más de 2.000 metros de cableado del sistema de videovigilancia, y
"delincuentes han cortado cables, lanzado piedras, incluso disparado a las cámaras" [2]. En Bogotá, la Contraloría
Distrital encontró 719 cámaras fuera de servicio en 2022 por robo de cableado, vandalismo y obras [3]. En Armenia,
la Alcaldía denunció en 2026 el hurto de "más de seis kilómetros" de cable y de tarjetas controladoras de la red de
semáforos [5]. Esa fuente no habla de cámaras, pero muestra que la infraestructura electrónica urbana de la
ciudad sufre el mismo tipo de daño.

Además del daño físico existe un sabotaje más difícil de notar: **tapar, desenfocar o girar la cámara**. La
cámara sigue encendida y transmitiendo, pero la imagen ya no sirve, y el problema puede pasar inadvertido hasta
que se necesitan las grabaciones. Con cientos de cámaras por operador, revisar cada imagen a mano no es viable.
Este proyecto construye un sistema que clasifica automáticamente cada imagen de una cámara en **cuatro clases:
normal, desenfocada, girada y tapada**, y reporta también la versión binaria (normal frente a saboteada), que es
la que dispara una alerta.

Siguiendo la guía del proyecto, el énfasis no está en obtener una precisión alta en condiciones ideales, sino en
**comparar dos metodologías y diagnosticar sus límites**. Se construyen en paralelo un módulo clásico (descriptores
matemáticos y reglas con umbrales calibrados) y un módulo de aprendizaje profundo (transfer learning con
MobileNetV2), y ambos se someten a una auditoría de robustez con cambios de luz, ruido y compresión, que son los
ataques que puede sufrir una cámara real.

**Datos.** No se encontró un dataset público regional de cámaras saboteadas: se buscó en datos.gov.co y en
repositorios académicos, y no hay cámaras web públicas de Armenia o Salento que puedan usarse. Por eso, como se
acordó con la docente, **el método se desarrolló y entrenó con el dataset de la Universidad de Bremen** [6], que
se usó para entender el problema y construir el sistema. El método no depende de esa ciudad y se aplica igual a un
dataset regional. La componente regional se evalúa con cámaras CCTV públicas de Medellín [7], a las que se aplica
sabotaje sintético, y con una captura propia en Armenia (sección 3).
