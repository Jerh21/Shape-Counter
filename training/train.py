import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset import MyShapesDataset
import matplotlib.pyplot as plt
import os
import time
from config import MODEL_TYPE
import warnings



warnings.filterwarnings("ignore", category=FutureWarning)



# === Importar el modelo correcto según config ===
if MODEL_TYPE == "resnet":
    from models.resnet_model import ShapeCounterResNet as ModelClass
else:
    from models.cnn import ShapeCounterCNN as ModelClass


def run_training():
    print(f"\nIniciando entrenamiento del modelo ({MODEL_TYPE.upper()})")
    start_time = time.time()

    # === Transformaciones ===
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # === Datasets y loaders ===
    train_dataset = MyShapesDataset("data/train", transform=train_transform)
    val_dataset   = MyShapesDataset("data/val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # === Modelo, pérdida, optimizador ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass().to(device)

    model_name = f"shape_counter_{MODEL_TYPE}"
    model_path = f"models/{model_name}.pth"

    if os.path.exists(model_path):
        opt = input(f"Modelo anterior encontrado ({model_name}). ¿Continuar entrenamiento? [s/N]: ").strip().lower()
        if opt == "s":
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"Pesos cargados desde {model_path}")
        else:
            print("ℹ️ Entrenamiento nuevo iniciado (pesos aleatorios).")
    else:
        print("ℹ️ No hay modelo previo. Se iniciará desde cero.")

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # === Entrenamiento ===
    num_epochs = 5
    os.makedirs("models", exist_ok=True)

    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(imgs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    output = model(imgs)
                    loss = criterion(output, labels)
                    val_loss += loss.item() * imgs.size(0)
            val_loss /= len(val_loader.dataset)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Tiempo transcurrido: {elapsed_str}")

            if (epoch + 1) % 25 == 0:
                torch.save(model.state_dict(), f"models/checkpoint_{MODEL_TYPE}_epoch{epoch+1}.pth")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"models/best_shape_counter_{MODEL_TYPE}.pth")

        torch.save(model.state_dict(), f"models/shape_counter_{MODEL_TYPE}.pth")
        print(f"\nEntrenamiento finalizado. Modelo guardado en models/shape_counter_{MODEL_TYPE}.pth")

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido. Guardando el modelo actual...")
        torch.save(model.state_dict(), f"models/last_shape_counter_{MODEL_TYPE}.pth")
        print(f"Modelo guardado como models/last_shape_counter_{MODEL_TYPE}.pth")

    # Guardar modelo final
    torch.save(model.state_dict(), model_path)
    print(f"\nEntrenamiento finalizado. Modelo guardado en {model_path}")

    # Guardar gráfica
    os.makedirs("output/logs", exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title(f"Pérdida por Época ({MODEL_TYPE.upper()})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/logs/loss_plot_{MODEL_TYPE}.png")
    print(f"📊 Gráfico de pérdidas guardado en output/logs/loss_plot_{MODEL_TYPE}.png")