# training
                                                            
Este módulo contiene la lógica principal para entrenar los modelos del proyecto. Se encarga de preparar los datos, configurar el modelo (CNN o ResNet), optimizarlo utilizando los ejemplos del dataset, monitorear su desempeño durante las épocas, y guardar los pesos entrenados. Es el corazón del proceso de aprendizaje automático en el sistema.



___

## Estructura del proyecto

📂training
 ┣ 📜train.py

___


## train.py (Corazón del entrenamiento)

Este script concentra toda la lógica necesaria para entrenar el modelo, ya sea una CNN personalizada o una arquitectura ResNet.

**Funcionalidad:**
- Preparar los datos y aplicar transformaciones
- Crear el modelo según la configuración (`cnn` o `resnet`)
- Definir la función de pérdida y el optimizador
- Ejecutar el ciclo de entrenamiento por varias épocas
- Evaluar el rendimiento en un set de validación
- Guardar el mejor modelo alcanzado
- Generar gráficas de pérdida para seguimiento del aprendizaje


___
```python 
def run_training():
```

- - Esta es la única función principal en `train.py`. Entrena el modelo paso a paso. Incluye desde preparar las imágenes, cargar el modelo, hacer las predicciones, calcular el error, y guardar los resultados.

## Explicación:
### 1. Carga de configuraciones y transformaciones
Primero define cómo se deben preparar las imágenes antes de mandarlas al modelo:

- Gira un poco las imágenes.
- Las reescala a 224x224 píxeles.
- Las convierte a tensores y las normaliza.

**Nota:**  
El modelo espera imágenes del mismo tamaño y con ciertos valores.

---

### 2️. Carga del dataset
Usa `MyShapesDataset` para cargar dos carpetas:

- `train` → imágenes para entrenar.
- `val` → imágenes para evaluar mientras se entrena.

Las pone en un `DataLoader` ( reparte las imágenes en lotes).

---

### 3️. Selección del modelo
Carga `config.py` (`cnn` o `resnet`).

Revisa si ya existe un modelo previamente entrenado (en `models/`) y pregunta si desea continuar desde ahí o empezar de cero.

---

### 4️. Define cómo aprenderá el modelo

- **Criterio (loss):** usa `SmoothL1Loss`, que castiga los errores de predicción.
- **Optimizador:** usa `Adam`, que ajusta los “pesos” del modelo para que aprenda.
- **Scheduler:** si el modelo no mejora, baja automáticamente la velocidad de aprendizaje.

---

### 5️. Bucle de entrenamiento (`for epoch in range(num_epochs):`)
Esto se repite varias veces para que el modelo aprenda mejor con cada vuelta (época).

#### a) Fase de entrenamiento

- Pasa las imágenes al modelo.
- Compara lo que predijo con lo correcto (etiquetas).
- Calcula el error.
- Ajusta sus pesos para intentar equivocarse menos la próxima vez.

#### b) Fase de validación

- Prueba con imágenes nuevas que no ha visto.
- Solo calcula el error, no aprende en esta fase.
- Sirve para saber si el modelo está mejorando o no.

---

### 6️. Guardado del modelo

- Si el modelo obtiene un error (`val_loss`) mejor que antes, guarda una copia como:

```bash
models/best_shape_counter_cnn.pth  # o resnet.pth
```

- También guarda un modelo cada 25 épocas como checkpoint:

```bash
models/checkpoint_cnn_epoch25.pth
```

- Al final del entrenamiento, guarda el modelo final con su nombre:

```bash
models/shape_counter_cnn.pth
```

---

### 7️. Soporte para interrupciones (`Ctrl+C`)
Si se corta el entrenamiento manualmente o por error (como me paso millones de veces), guarda una copia del modelo actual como:

```bash
models/last_shape_counter_cnn.pth
```

---

### 8️. Gráfico de pérdidas
Al terminar, genera una imagen (`.png`) que muestra cómo bajó el error durante las épocas.

- Eje X → número de época  
- Eje Y → cuánto se equivocaba el modelo (loss)

Se guarda como:

```bash
output/logs/loss_plot_cnn.png
```
