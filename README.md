# Proceso
En este repositorio se realiza la implementación de un modelo para la predicción de impago. Teniendo en cuenta múltiples características transaccionales de cada cliente, se busca determinar si este pagará o no el próximo mes.

El marco de trabajo se detalla a continuación:

1. Entendimiento del problema
2. Análisis de la calidad de los datos
3. Corrección de la calidad de los datos
4. Análisis exploratorio
5. Ingeniería de características
6. Selección de variables
7. Experimentos para la selección del mejor modelo (selección de métricas, búsqueda de hiperparámetros)
8. Selección del modelo
9. Refactorización del código (modularización del código, construcción de clases, docstring y anotaciones)
10. Construcción de pipelines de datos y entrenamiento de modelo
11. Model registry.

## Notas
Se construyen librerías personalizadas que permiten un rápido reentrenamiento y puesta en producción del mejor modelo. Esta implementación no se encuentra restringida solamente al conjunto de datos en cuestión (default-of-credit-card-clients). Si no que, cambiando los parámetros de configuración en el archivo metadata.yaml y la ruta donde se almacenan los datos, se pueden usar para la construcción de modelos que apunten a diferentes objetivos.    

# Organización del repositorio

```
├── analisis.ipynb               <- Consolida todo el marco de trabajo detallado anteriormente. Usa los módulos y las funciones que se describen a continuación. Además, contiene comentarios y detalla el proceso.
│
├── metadata.yaml                <- Valores de configuración de la base de datos usada para el modelamiento.
│
├── registry                     <- Carpeta con binarios de modelos, escalares, pca, etc.
│   ├── data_pipeline            <- Binarios del pipeline de datos.
│   └── model_pipeline           <- Binario del pipeline del modelo.
│
├── src                          <- Módulos para limpieza de datos, exploración de datos y ajuste de modelos.
│   ├── data_pipeline            <- Módulos del pipeline de datos.
│   │   ├── __init__.py          <- Contiene la clase principal del pipeline de datos que permite ejecutarlo.
│   │   ├── data_cleansing.py    <- Funciones usadas en la limpieza de datos.
│   │   └── feature_engineering.py <- Funciones usadas en la ingeniería de características.
│   ├── model_pipeline           <- Módulos del pipeline del modelo.
│   │   ├── __init__.py          <- Contiene la clase principal del pipeline del modelo que permite ejecutarlo.
│   │   ├── feature_selection.py <- Funciones usadas en la selección de características.
│   │   └── model_selection.py   <- Módulo para entrenar y evaluar modelos.
│   ├── exploratory.py           <- Funciones para la visualización de datos.
│   ├── utils                    <- Funciones para el guardado y la lectura de variables en el registry.
│   └── __init__.py              <- Archivo que se lee primero al importar el folder datasets.
│
└── requirements.txt             <- Archivo con las versiones de los paquetes necesarios. 

                           
```