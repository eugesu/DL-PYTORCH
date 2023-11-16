# DL-PYTORCH

# Eugenio Sánchez | Noviembre 2023
# IIT | Comillas - ICAI

Conjunto de cuadernos de Jupyter que permiten:
  - Analizar el dataset público MNIST para entender la complejidad del problema y sus características principales
  - Entrenar un MLP para predecir los dígitos del dataset público MNIST
  - Entrenar una CNN para predecir los dígitos del dataset público MNIST
  - Generar un dataset artificial con dígitos impresos utilizando las fuentes instaladas del ordenador
  - Ejecutar el MLP y la CNN ajustados en MNIST para ver cómo generalizan
  - Reentrenar la CNN con los datos de ESU23 (transfer learning)

Para poder ejecutar los notebooks es necesario disponer de las carpetas de datos y modelos.

La carpeta de data se construye ejecutando:
 -  1_descarga_datasets_MNIST -> descarga el dataset de MNIST (dígitos manuscritos) en la carpeta data
 -  5_genera_imagenes_sinteticas_ESU23* ->   genera dataset ESU23 (dígitos impresos) en la carpeta data

 *El notebook 5_genera_imagenes_sinteticas_ESU23.ipynb requiere adaptarlo a las fuentes que se tengan instaladas en el ordenador, hay código para poderlo hacer fácilmente.
