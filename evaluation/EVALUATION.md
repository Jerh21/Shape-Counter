# evaluation
                                                            
Este módulo agrupa las funcionalidades necesarias para evaluar el rendimiento del modelo una vez entrenado, así como para realizar predicciones sobre imágenes externas.

___

## Estructura del proyecto

📂evaluation
 ┣ 📜metrics.py
 ┗ 📜predict.py

__
## metrics.py

Este script se encarga de calcular las métricas cuantitativas del modelo en el conjunto de prueba (`data/test`).

**Funcionalidad:**
- Evalúa el modelo usando **MAE** y **R²** sobre las predicciones.
- Permite elegir entre el modelo **CNN** o **ResNet**, dependiendo de `config.py`.
- Muestra resultados tanto por clase como totales.

**Métricas calculadas:**

- `MAE` (Mean Absolute Error): error promedio absoluto por clase.
- `R²` (coeficiente de determinación): mide qué tan bien el modelo explica la variabilidad del conteo real.


## predict.py (parte vital del proyecto)

Este script permite utilizar el modelo entrenado para predecir el conteo de figuras en imágenes externas, ubicadas en la carpeta `input/`.

**Funcionalidad:**
- Carga el modelo especificado en `config.py` (`cnn` o `resnet`).
- Procesa todas las imágenes en la carpeta `input/`.
- Realiza predicciones y:
  - Anota el resultado en la imagen.
  - Guarda el conteo en un archivo `.txt`.
  - Genera imágenes con las predicciones visuales en `output/`.

```python 
def predict_counts(img_path):
```

- - Esta función toma una imagen y le pregunta al modelo cuántas figuras hay de cada tipo (círculo, triángulo,      cuadrado).


    - Parámetro: `img_path` Ruta de la imagen a análizar por el modelo.

    - Fases:
        1. Abre la imagen y la convierte a blanco y negro (escala de grises).

        2. La prepara para que el modelo pueda entenderla (la convierte a tensor, la normaliza, etc.).

        3. Se la pasa al modelo.

        4. El modelo devuelve tres números: uno para cada tipo de figura.

        5. Los números se redondean y se aseguran de que no sean negativos.
    
    ```python 
        return: [3, 2, 5]  # → 3 círculos, 2 triángulos, 5 cuadrados
    ```


```python 
def draw_prediction(image, counts):
```

- - Esta función escribe los resultados del conteo sobre la imagen, usando texto de colores..


    - Parámetro: `image` La imagen resultante del conteo.
    - Parámetro: `counts` Número de figuras.

    - Fases:
        1. Convierte la imagen a color (para que se puedan ver los textos en colores).

        2. Usa una fuente de letra y empieza a dibujar.

    Retorna la imagen con el conteo de figuras escrito en colores:
    - Azul para círculos
    - Verde para triángulos.
    - Rojo para cuadrados.

```python 
def run_prediction(best=False):
```

- - Esta es la función principal. Es la que recorre todas las imágenes en la carpeta input/ y predice cada una.


    - Parámetro: `best` Se refiere al modelo que utilizaremos, false para utilizar el último disponible, true para utilizar el modelo con mejores resultados en el entrenamiento.

    - Fases:
        1. Decide si cargar el modelo "normal" (`shape_counter_cnn.pth`) o el "mejor" (`best_shape_counter_cnn.pth`), dependiendo del parámetro `best`.

        2. Revisa una por una las imágenes dentro de input/.
        
        3. Para cada imagen:
             1. Usa predict_counts para predecir el conteo.

             2. Dibuja los resultados con `draw_prediction`.

             3. Guarda la imagen nueva en output/.

             4. Crea un archivo `.txt` con los conteos.

    - Retorna: 
        1. Imagen anotada con texto.

        2. Archivo .txt con el conteo.


```python 
def get_predict_function():
```

- - Esta función no se usa directamente al predecir. Se incluye como apoyo para que otras funciones (dentro de `metrics.py` por ejemplo) puedan usar la función `predict_counts()` sin repetir código.