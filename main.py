import time
from dataset.split import split_dataset

def menu():
    print("\nContador de Figuras")
    print("1. Generar imágenes sintéticas")
    print("2. Dividir dataset en train/val/test")
    print("3. Entrenar modelo")
    print("4. Predecir usando el mejor modelo guardado")
    print("5. Salir")


def main():
    while True:
        menu()
        opcion = input("\nElegir una opción (1-5): ").strip()
        if opcion == "1":
            from dataset.generator import run_generation
            run_generation()
        elif opcion == "2":
            from dataset.split import split_dataset
            split_dataset()
        elif opcion == "3":
            from training.train import run_training
            run_training()
        elif opcion == "4":
            from evaluation.predict import run_prediction
            run_prediction(best=True)
        elif opcion == "5":
            print("Gracias")
            time.sleep(1)
            break
        else:
            print("No válida.")

if __name__ == "__main__":
    main()