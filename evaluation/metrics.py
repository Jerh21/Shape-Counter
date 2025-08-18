import os
import json
import numpy as np
from glob import glob
from sklearn.metrics import r2_score

def load_labels_and_predictions(data_folder, prediction_func):
    """
    Ejecuta el modelo sobre cada imagen en data_folder
    y devuelve labels y predicciones como listas de 3 elementos (C, T, Q).
    
    prediction_func debe recibir una ruta de imagen y devolver [círculos, triángulos, cuadrados]
    """
    image_files = sorted(glob(os.path.join(data_folder, "*.jpg")))
    labels = []
    preds = []

    for img_path in image_files:
        json_path = img_path.replace(".jpg", ".json")
        if not os.path.exists(json_path):
            continue

        # Cargar label real
        with open(json_path, "r") as f:
            gt = json.load(f)["counts"]
        labels.append([gt["circle"], gt["triangle"], gt["square"]])

        # Ejecutar modelo
        pred = prediction_func(img_path)
        preds.append(pred)

    return np.array(labels), np.array(preds)

def compute_mae(labels, preds):
    mae = np.abs(labels - preds).mean(axis=0)
    total = np.abs(labels - preds).mean()
    return {
        "circle_mae": round(mae[0], 2),
        "triangle_mae": round(mae[1], 2),
        "square_mae": round(mae[2], 2),
        "total_mae": round(total, 2)
    }

def run_metrics(data_folder, prediction_func):
    """
    Corre evaluación sobre una carpeta con imágenes y JSONs.
    prediction_func debe ser una función que reciba img_path y retorne [c,t,q]
    """
    print(f"\n📊 Evaluando modelo sobre: {data_folder}")
    labels, preds = load_labels_and_predictions(data_folder, prediction_func)
    
    if len(labels) == 0:
        print("❌ No se encontraron imágenes con JSON.")
        return

    metrics = compute_mae(labels, preds)
    r2_metrics = compute_r2(labels, preds)

    print("\n📈 Resultados MAE:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n📊 Resultados R²:")
    for k, v in r2_metrics.items():
        print(f"{k}: {v}")

def compute_r2(labels, preds):
    r2 = r2_score(labels, preds, multioutput='raw_values')
    total_r2 = r2_score(labels, preds)
    return {
        "circle_r2": round(r2[0], 2),
        "triangle_r2": round(r2[1], 2),
        "square_r2": round(r2[2], 2),
        "total_r2": round(total_r2, 2)
    }