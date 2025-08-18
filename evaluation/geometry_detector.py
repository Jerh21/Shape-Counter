import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

def detect_and_draw_shapes(pil_img):
    """
    Detecta figuras geométricas y las anota en color.
    Devuelve una imagen anotada y un diccionario con conteos.
    """
    counts = {"circle": 0, "triangle": 0, "square": 0}

    # Convertir PIL a OpenCV
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)

        if len(approx) == 3:
            shape = "triangle"
            color = (0, 255, 0)
        elif len(approx) == 4:
            aspect_ratio = w / float(h)
            if 0.9 < aspect_ratio < 1.1:
                shape = "square"
                color = (255, 0, 0)
            else:
                continue
        elif len(approx) > 6:
            shape = "circle"
            color = (0, 0, 255)
        else:
            continue

        cv2.drawContours(img, [cnt], -1, color, 2)
        cv2.putText(img, shape, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        counts[shape] += 1

    # Convertir de vuelta a PIL
    annotated = Image.fromarray(img)
    return annotated, counts

def run_geometry_detection():
    INPUT_FOLDER = "input"
    OUTPUT_FOLDER = "output"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    found = False
    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        found = True
        img_path = os.path.join(INPUT_FOLDER, fname)
        image = Image.open(img_path)
        annotated, counts = detect_and_draw_shapes(image)

        # Guardar imagen anotada
        annotated.save(os.path.join(OUTPUT_FOLDER, fname))

        # Guardar resultados .txt
        with open(os.path.join(OUTPUT_FOLDER, fname + ".txt"), "w") as f:
            for shape in ["circle", "triangle", "square"]:
                f.write(f"{shape}: {counts[shape]}\n")

        print(f"\nArchivo: {fname}")
        for shape in ["circle", "triangle", "square"]:
            print(f"{shape}: {counts[shape]}")

    if not found:
        print("No se encontraron imágenes en 'input/'.")
    else:
        print("Detección geométrica completada.")