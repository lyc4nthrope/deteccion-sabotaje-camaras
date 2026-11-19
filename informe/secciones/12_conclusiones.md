# 12. Conclusiones y trabajo futuro

## 12.1 Conclusiones por objetivo

**Objetivo general.** Se construyeron y auditaron en paralelo un módulo clásico y uno de aprendizaje profundo para
detectar cámaras tapadas, desenfocadas y giradas. La conclusión principal es que **ningún módulo es más robusto en
todo**: empatan en condiciones limpias, la CNN resiste mejor la luz y el ruido leve en la cámara que conoce, y el
clásico generaliza mucho mejor a una cámara nueva y es el único cuyo fallo se explica y se corrige con un filtro
medido (sección 10).

1. **Conjunto de datos sin fugas: cumplido.** La revisión del dataset de Bremen encontró 751 archivos duplicados
   (tabla 3.1), entre ellos que la clase girada de la ubicación 1 tiene 135 imágenes distintas y no 600, y que las
   tapadas de la ubicación 2 son copias de la ubicación 1. La división por bloques de 12 imágenes consecutivas, con
   2 descartadas a cada lado de cada frontera, dejó 1930 imágenes de entrenamiento, 352 de validación y 327 de
   prueba (tabla 3.2), y la ubicación 2 quedó reservada como cámara no vista (1163 imágenes evaluadas). Queda un
   sesgo que la división no puede quitar: cada clase se grabó en días distintos (sección 10.4).
2. **Módulo clásico: cumplido.** Ocho descriptores (nitidez, contraste, entropía, color en HSV, coincidencias ORB y
   ángulo por RANSAC) y un árbol de reglas con umbrales calibrados solo en entrenamiento dan F1 macro de 0,954 y
   0,5 % de falsas alarmas en prueba, y 0,920 y 0,1 % en la cámara no vista, con 12,6 ms por cuadro (sección 7.1).
   ORB se eligió frente a SIFT y KAZE por velocidad con una separación equivalente (tabla 5.3).
3. **Módulo de aprendizaje profundo: cumplido, con una limitación central.** MobileNetV2 con la base congelada,
   tasa 1e-2 y lote 32, elegida con la pérdida de validación entre 7 corridas (`results/cnn/hiperparametros.csv`),
   da F1 macro de 0,967 y 0 % de falsas alarmas en prueba. Hay una señal leve de sobreajuste después de la época 3,
   que la parada temprana controla al restaurar los pesos de esa época (`results/cnn/lr1e-2_b32_frozen/history.csv`).
   En la cámara no vista cae a 0,580 (`results/cnn/lr1e-2_b32_frozen/evaluacion/resumen.csv`): aprende la cámara de
   entrenamiento, no solo el sabotaje.
4. **Auditoría de robustez: cumplida en la parte artificial, pendiente en la física.** Los dos módulos se evaluaron
   sobre las mismas 6213 imágenes (327 × 19 condiciones), con matrices cruzadas, ROC por condición y F1 frente a
   severidad (sección 8). La CNN tiene mejor F1 en 10 condiciones y el clásico en 4; los dos mantienen el AUC
   binaria en ≥ 0,96 en 36 de 38 combinaciones, así que fallan más en el **tipo** de sabotaje que en su
   **detección**. La degradación física con luz real (T75) no se hizo.
5. **Diagnóstico y corrección: cumplido para el clásico, parcial para la CNN.** Para el clásico, la fórmula
   4,7 + 8,94 σ² predice con menos de 2 % de error cómo el ruido infla el Laplaciano (tabla 9.1), y cada ataque
   tiene un filtro medido: la mediana recupera la sal y pimienta de niveles 1 y 2 (F1 0,98) y la gamma inversa
   lleva gamma 3 de 0,599 a 0,871 (tabla 9.4). La auditoría también mostró que una corrección a ciegas daña: CLAHE
   baja el F1 limpio de 0,954 a 0,323. Para la CNN, síntomas y dirección del error están medidos y la causa de la sal
   y pimienta se confirmó al aplicar la mediana antes de la red (F1 0,221 → 0,810, tabla 9.6); las demás causas
   siguen siendo hipótesis: Grad-CAM (T67) da mapas centrales y difusos que no las resuelven.
6. **Generalización regional: cumplido en parte.** La prueba en cámaras de Medellín se hizo con sabotaje sintético
   (sección 3.5, tabla 3.3) y repite el resultado de la cámara no vista: F1 0,782 para el clásico y 0,404 para la
   CNN, con falsas alarmas de 13,2 % y 14,3 %. Es una prueba sintética, ajustada midiendo el propio clásico, así
   que favorece a ese módulo. **La captura propia en Armenia no se hizo**, y con ella la muestra de sabotaje real
   de la región y la tapa con la mano. La otra medida de generalización es la segunda cámara de Bremen (F1 0,920
   frente a 0,580).

**Lección de método.** La prueba en la misma cámara habría llevado a una conclusión equivocada: con F1 de 0,954 y
0,967, la CNN parecía igual o mejor. Solo la cámara no vista mostró que las reglas, al compararse contra referencias
de la misma cámara, generalizan, y que cualquier modelo que aprende valores absolutos de una sola cámara —la CNN y
también KNN, SVM y Naive Bayes sobre los descriptores clásicos (sección 7.3)— no lo hace.

## 12.2 Trabajo futuro

1. **Diagnosticar por qué la CNN falla en Medellín (T76).** La prueba mostró el síntoma (deja pasar como normal el
   79 % de las giradas) pero no la causa. Queda aplicar Grad-CAM e intervenciones a esas imágenes, y repetir la
   prueba con sabotaje que no se haya ajustado contra el módulo clásico.
2. **Captura propia en Armenia y auditoría física (T81, T75).** El protocolo y el ayudante de captura ya existen
   (`src/capture.py`); faltan las fotos. Con ellas se mide la tapa con la mano, que Bremen no tiene (sección 11.3),
   y se compara la luz real con la perturbación artificial equivalente: si el oscurecimiento por código predice la
   luz baja real.
3. **Explicabilidad de la CNN más fina que Grad-CAM.** Grad-CAM (T67) se hizo sobre imágenes sin perturbar y su mapa
   de 7×7 es demasiado grueso para ver si la red se fija en el sabotaje o en el contexto del día de grabación. Queda
   correrlo sobre el conjunto degradado y probar métodos de mayor resolución para confirmar o descartar las hipótesis de la sección 10.4 y de la sección 9.5, y
   medir CLAHE o el filtro bilateral antes de la red (la mediana ya se midió, tabla 9.6).
4. **Validar las correcciones en validación.** La elección del bilateral para el ruido gaussiano y la decisión de
   dejar la mediana siempre activa se tomaron en prueba (sección 11.2) y deben confirmarse en validación.
5. **Sistema híbrido (propuesta, no medida).** Los resultados sugieren combinar los módulos en lugar de elegir uno:
   - un **estimador de la perturbación** (ruido, brillo, saturación) que active solo el filtro que corresponde,
     porque un filtro a ciegas puede hacer más daño que el ataque (sección 9.4);
   - el **clásico como detector principal** de normal frente a saboteada, por su generalización a cámaras nuevas,
     sus pocas falsas alarmas y su calibración de 8 referencias;
   - la **CNN como segunda opinión sobre el tipo** de sabotaje solo en cámaras de las que haya imágenes de
     entrenamiento, donde resiste mejor la luz y el ruido leve.

   Esta arquitectura es una propuesta derivada de las secciones 8 a 10; ninguna de sus combinaciones se ha
   evaluado, y la robustez de la CNN en una cámara nueva no está medida.
