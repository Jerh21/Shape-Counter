import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from config import MODEL_TYPE
from evaluation.geometry_detector import detect_and_draw_shapes
import warnings



warnings.filterwarnings("ignore", category=FutureWarning)

# === Importar modelo adecuado ===
if MODEL_TYPE == "resnet":
    from models.resnet_model import ShapeCounterResNet as ModelClass
else:
    from models.cnn import ShapeCounterCNN as ModelClass

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

# === Transformaciones ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Cargar modelo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelClass().to(device)
model.eval()

def predict_counts(img_path):
    """Predice [círculos, triángulos, cuadrados] para una imagen"""
    image = Image.open(img_path).convert("L")  # Escala de grises
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.round(output).clamp(min=0).cpu().numpy()[0].astype(int)
        return pred.tolist()

def draw_prediction(image, counts):
    """Dibuja los conteos sobre la imagen"""
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"circle": "blue", "triangle": "green", "square": "red"}

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    x, y = 10, 10
    for shape, count in counts.items():
        draw.text((x, y), f"{shape.capitalize()}: {count}", fill=colors[shape], font=font)
        y += 25
    return image

def run_prediction(best=False):
    suffix = f"{MODEL_TYPE}"
    model_path = f"models/best_shape_counter_{suffix}.pth" if best else f"models/shape_counter_{suffix}.pth"
    model_desc = f"mejor modelo ({MODEL_TYPE})" if best else f"último modelo entrenado ({MODEL_TYPE})"
    print(f"Ejecutando predicción usando {model_desc}...")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"No se encontró el modelo: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    found = False
    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        found = True
        img_path = os.path.join(INPUT_FOLDER, fname)
        original_img = Image.open(img_path).convert("RGB")

        counts_list = predict_counts(img_path)
        counts = {
            "circle": counts_list[0],
            "triangle": counts_list[1],
            "square": counts_list[2]
        }

        print(f"\nImagen: {fname}")
        for shape in ["circle", "triangle", "square"]:
            print(f"{shape.capitalize()}: {counts[shape]}")

        # Imagen anotada
        img_with_text = draw_prediction(original_img, counts)
        img_with_text.save(os.path.join(OUTPUT_FOLDER, fname))

        # Conteo en .txt
        with open(os.path.join(OUTPUT_FOLDER, fname + ".txt"), "w") as f:
            for shape in ["circle", "triangle", "square"]:
                f.write(f"{shape}: {counts[shape]}\n")

    if not found:
        print("No se encontraron imágenes en 'input/'.")
    else:
        print(f"\nPredicción finalizada. Resultados guardados en '{OUTPUT_FOLDER}/'.")

# Exponer para métricas
def get_predict_function():
    return predict_counts