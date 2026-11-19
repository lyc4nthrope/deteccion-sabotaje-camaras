# 2. Objetivos

## Objetivo general

Detectar el sabotaje de cámaras de vigilancia (tapar, desenfocar o girar la cámara) con dos enfoques en paralelo,
procesamiento clásico y aprendizaje profundo, para auditar, comparar y diagnosticar sus límites operativos ante
perturbaciones del entorno como cambios de luz y ruido.

## Objetivos específicos

1. **Construir un conjunto de datos sin fugas:** depurar el dataset de Bremen (duplicados y copias entre clases)
   y dividirlo en entrenamiento, validación y prueba por bloques de tiempo, con una segunda cámara reservada como
   prueba de generalización (sección 3).
2. **Implementar el módulo clásico:** descriptores matemáticos de nitidez, uniformidad, color y geometría, y
   reglas con umbrales calibrados y justificados; medir el tiempo por cuadro y la tasa de falsos positivos
   (sección 5).
3. **Implementar el módulo de aprendizaje profundo:** transfer learning con MobileNetV2, hiperparámetros elegidos
   con evidencia y curvas de pérdida y accuracy sin sobreajuste severo (sección 6).
4. **Auditar la robustez de ambos módulos:** 6 perturbaciones con 3 niveles cada una (la degradación física con
   luz real no se hizo, sección 11); matrices de confusión cruzadas, curvas ROC y F1 en condiciones críticas (sección 8).
5. **Diagnosticar y corregir cada fallo:** explicar con la teoría del píxel qué descriptor o capa falla ante cada
   perturbación, proponer el filtro correctivo exacto y medir la mejora (sección 9).
6. **Probar la generalización regional:** evaluar en cámaras de Medellín, hecho con sabotaje sintético, y en una
   captura propia en Armenia, que no se realizó (secciones 3, 8 y 11).
