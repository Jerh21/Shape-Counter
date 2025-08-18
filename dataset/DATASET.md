# dataset
                                                            
Es la parte encargada de generar los recursos utilizados en el modelo, es decir, las imagenes que utilizaremos ya sea para probar o entrenar el modelo. 

>[!TIP]
>
>El generador de imagenes permite diversas configuraciones.


___

## Estructura del proyecto

📂dataset
 ┣ 📜config.py
 ┣ 📜dataset.py
 ┣ 📜generator.py
 ┗ 📜split.py

___

## config.py

Son diferentes parámetros que definen cómo se generan las imágenes del dataset. Funciona como una configuración centralizada para la generación de datos sintéticos.

**Parámetros más importantes:**
- `IMG_SIZE`: Resolución de las imágenes generadas (Actulmente. `(224, 224)`).
- `MAX_SHAPES`: Cantidad máxima de figuras de cada tipo.
- `SHAPE_TYPES`: Tipos de figuras que se incluirán (Son 3 en este caso: `circle`, `triangle`, `square`).
- `NUM_IMAGES`: Cuántas imágenes se generarán, definible por usuario al momento de ejecutar la generación.

___

## generator.py

Es el principal encargado de generar las imágenes y producir sus respectivas etiquetas `.json`, aquí se maneja toda la mayor parte de la lógica detras de la generación.

**Responsabilidades:**
- Crear imágenes con formas distribuidas aleatoriamente.
- Guardar etiquetas con el conteo real por clase.
- Evitar solapamientos severos (dependiendo de configuración).
- Guardar las imágenes en `data/all/`.

**Salida que espera:**
- Archivos `.jpg` y su etiqueta `.json` con estructura:
  ```json
  { "circle": 9, "triangle": 10, "square": 7 }

El archivo `.json` es de suma utilidad, ya sea para entrenar el modelo como también para que el usuario pueda verificar cuántas figura tiene la imagen.

## split.py

Divide las imagenes generadas por `generator.py` en: entrenamiento, validación y prueba, siguiendo las proporciones estándar:

-
  -  train: 70%
  -  val: 15%
  -  test: 15%


## dataset.py

`dataset.py` es usado por train.py para cargar y procesar datos en PyTorch.

Implementa la clase `MyShapesDataset`, que extiende `torch.utils.data.Dataset`.

**Funcionalidad principal:**

- Leer imágenes desde disco y sus etiquetas asociadas (.json).
- Aplica transformaciones definidas por el usuario (transforms).
- Devuelve tuplas (imagen_tensor, etiqueta_tensor) listas para ser usadas en entrenamiento o validación.