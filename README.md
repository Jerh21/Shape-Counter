# SHAPE_COUNTER
 ____  _   _    _    ____  _____    ____ ___  _   _ _   _ _____ _____ ____  

/ ___|| | | |  / \  |  _ \| ____|  / ___/ _ \| | | | \ | |_   _| ____|  _ \ 

\___ \| |_| | / _ \ | |_) |  _|   | |  | | | | | | |  \| | | | |  _| | |_) |

 ___) |  _  |/ ___ \|  __/| |___  | |__| |_| | |_| | |\  | | | | |___|  _ < 

|____/|_| |_/_/   \_\_|   |_____|  \____\___/ \___/|_| \_| |_| |_____|_| \_\

                                                                            
Contador de figuras geométricas en imágenes, usando redes neuronales convolucionales.

___
#### Creador: 
- Josué Esaú Ham Romero             20191001154 
___
## Objetivo

Este proyecto tiene como objetivo principal entrenar un modelo que sea capaz de reconocer cuántas figuras geométricas de cada tipo (círculo, triángulo y cuadrado) hay en una imagen generada.

El sistema se entrena a partir de imágenes con etiquetas generadas automáticamente, lo que permite un entrenamiento sin intervención manual. Se incluyen dos arquitecturas: una CNN personalizada y una versión alternativa con ResNet18.

___

## Herramientas utilizadas

- **Python 3.10+**
- **PyTorch** (entrenamiento y predicción)
- **Torchvision** (modelo ResNet)
- **NumPy, Matplotlib** (visualización y métricas)
- **PIL / Pillow** (carga y anotación de imágenes)

---

## 📁 Estructura del proyecto

📦shape_counter
 ┣ 📂data
 ┣ 📂dataset
 ┃ ┣ 📜config.py
 ┃ ┣ 📜dataset.py
 ┃ ┣ 📜generator.py
 ┃ ┗ 📜split.py
 ┣ 📂evaluation
 ┃ ┣ 📜geometry_detector.py
 ┃ ┣ 📜metrics.py
 ┃ ┗ 📜predict.py
 ┣ 📂input
 ┣ 📂models
 ┃ ┣ 📜cnn.py
 ┃ ┣ 📜resnet_model.py
 ┣ 📂output
 ┃ ┗ 📂logs
 ┣ 📂training
 ┃ ┗ 📜train.py
 ┣ 📜config.py
 ┗ 📜main.py

___

 ## Ejecución del proyecto

Para ejecutar el proyecto primero se debe instalar `requirements.txt`

`pip install -r requirements.txt`

Esto nos instala las librerias necesarias para la ejecución del proyecto

Una vez instalados los requerimientos, podemos ejecutar el programa `main.py`

`python main.py`


___

 ## Shape Counter

- ### config.py
Configuración del proyecto. Definiremos el tipo de modelo a utilizar cnn o resnet.

- ### main.py
Menú principal donde ejecutaremos todas las funcionalidades del proyecto (generación, entrenamiento y predicción).

- ### dataset/
Módulos para la creación del dataset (Se refiere a los recursos necesarios para el proyecto):

-
    -   `generator.py:` Es el encargado de generar las imágenes y etiquetas `.json`.
    -   `split.py:` Encargador de hacer la division del dataset en entrenamiento, validación y prueba.
    -   `config.py:` parámetros de generación (resolución, cantidad de figuras, etc.).
    -   `dataset.py:` clase Dataset para PyTorch.

- ### models/
Almacena los modelos de nuestro proyecto:

-
    -   `cnn.py:` red convolucional personalizada.
    -   `resnet_model.py:` adaptación de ResNet18 para regresión de conteos.

- ### training/
- - `train.py` Maneja la lógica de entrenamiento del modelo.

- ### evaluation/
Módulos para evaluar el rendimiento del modelo:

-
   -  `metrics.py:` calcula MAE y R² en el test set.

   -  `predict.py:` realiza predicciones sobre imágenes externas o sets completos.

   -  `geometry_detector.py:` método alternativo con OpenCV para detección basada en contornos (descontinuado).

- ### input/
Carpeta donde se colocan las imágenes que desean ser evaluadas por el modelo, permite multiple imagenes a la vez.

- ### output/
Resultados generados por el sistema. Cuando termina de evaluar nos dara la imagen evaluada, en ella tendra un conteo de la figura, tambien generara un .txt con la cantidad de figuras encontradas.

 - - Subcarpeta logs/ para guardar gráficas y métricas al momento de terminar el entrenamiento del modelo.

- ### data/
Dataset generado automáticamente, dividido en:

-
  - train (guarda el 70% de las imagenes) (Imagenes usadas en el entrenamiento)

  - val (guarda el 15% de las imagenes) (Imagenes usadas para evaluar mientras se entrena)

  - test (guarda el 15% de las imagenes) (Imagenes utilizadas para medir el rendimiento y acierto)

  - all (Nota: dataset original completo sin dividir)


- # Recomendaciones:

1. Generar imágenes sintéticas
   - Es necesario generar un dataset para poder entrenar el modelo, recomiendo generar entre 5,000 a 50,000 (o más) imagenes, estas imagenes pueden ser tomadas para probar el modelo 
  
2. Dividir dataset en train/val/test
   - Antes de entrenar el modelo es necesario tambien dividir el dataset en carpetas, para eso utilizaremos esta opción.

3. Entrenar modelo
   Aquí se entrenara el modelo utilizando el dataset.
    ```python
    # === Modelo, pérdida, optimizador ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass().to(device)
    ```

    **Se recomienda ampliamente cambiar esta opción a "cpu" en la linea de device si no se posee una gpu**

4. Predecir usando el mejor modelo guardado"

   >[!NOTE]
   >
   >El modelo `best_shape_counter_resnet` que ya viene incluido fue entrenado con 37500 imagenes, durante 5 horas usando cuda cores (GPU).
   >Alcanzó "Loss: 0.0505 - Val Loss: 0.0330 - Tiempo transcurrido: 04:33:24"
