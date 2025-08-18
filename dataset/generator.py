from PIL import Image, ImageDraw
import random
import math
import json
import os
from .config import *

def draw_square(draw, x, y, size):
    draw.rectangle([x, y, x + size, y + size], outline="black", width=2)
    return {
        "type": "square",
        "x": x, "y": y, "size": size
    }

def draw_circle(draw, x, y, size):
    radius = size // 2
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline="black", width=2)
    return {
        "type": "circle",
        "center": (x, y),
        "radius": radius,
        "size": size
    }

def draw_triangle(draw, x, y, size):
    height = int(size * math.sqrt(3) / 2)
    points = [
        (x, y - height // 2),
        (x - size // 2, y + height // 2),
        (x + size // 2, y + height // 2)
    ]
    draw.polygon(points, outline="black", width=2)
    return {
        "type": "triangle",
        "points": points,
        "size": size
    }

def generate_image(index, output_dir):
    image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    counts = {shape: 0 for shape in SHAPES}
    figures = []

    # Total aleatorio de figuras por imagen
    total_shapes = random.randint(5, 45)

    for _ in range(total_shapes):
        shape = random.choice(SHAPES)  # Tipo aleatorio
        size = random.randint(MIN_SIZE, MAX_SIZE)
        x = random.randint(size, IMG_WIDTH - size)
        y = random.randint(size, IMG_HEIGHT - size)

        if shape == "circle":
            figures.append(draw_circle(draw, x, y, size))
        elif shape == "square":
            figures.append(draw_square(draw, x, y, size))
        elif shape == "triangle":
            figures.append(draw_triangle(draw, x, y, size))

        counts[shape] += 1

    img_name = os.path.join(output_dir, f"shapes_{index:04d}.jpg")
    json_name = img_name.replace(".jpg", ".json")

    image.save(img_name, "JPEG", quality=95)
    with open(json_name, "w") as f:
        json.dump({"counts": counts, "figures": figures}, f, indent=2)

    return img_name, json_name

def run_generation():
    import os
    output_dir = input("Ruta de salida [por defecto: data/all]: ") or "data/all"
    num = input("¿Cuántas imágenes quieres generar? [por defecto: 1]: ") or "1"

    try:
        num = int(num)
        os.makedirs(output_dir, exist_ok=True)
        for i in range(num):
            generate_image(i, output_dir)
            print(f"[{i+1}/{num}] Imagen generada")
        print(f"\n✅ Se generaron {num} imágenes en '{output_dir}'")
    except Exception as e:
        print(f"❌ Error al generar imágenes: {e}")