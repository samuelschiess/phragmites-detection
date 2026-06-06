import os
import sys
import yaml
import torch
from ultralytics import YOLO

# Load best hyperparameters if availalble
def load_best_hyperparameters():
    pass

# Train the model
def main():
    print("==================================================")
    print("YOLO OBB Model Training")
    print("==================================================")

    # Check for GPU
    print("==================================================")
    print("Checking for NVIDIA GPU and CUDA availability...")
    print("==================================================")
    cuda_avail = torch.cuda.is_available()
    print(f"[*] CUDA Available: {cuda_avail}")

    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        print(f"[*] NVIDIA GPU Detected: {device_name}")
        device = 0
    else:
        print("[!] WARNING: No GPU found, running on CPU")
        response = input("Do you want to continue? (yes/no): ").lower().strip()
        if response not in ["yes", "y"]:
            print("Exiting training. Please run on a machine with an NVIDIA GPU for best performance.")
            sys.exit(0)
        device = "cpu"

    print("==================================================")

    # Load Model
    model_name = "yolo26n-obb.pt"
    print(f"[*] Loading pre-trained model: {model_name}")
    model = YOLO(model_name)
    print(f"[*] Model loaded successfully: {model_name}")

    # Load hyperparameters

    # Train the model
    print("[*] Starting model training...")
    model.train(
        data=os.path.join("datasets", "mvp_2", "data.yaml"),
        cfg="runs/obb/mvp_2/best_hyperparameters.yaml",
        epochs=100,
        optimizer="AdamW",
        device=device,
        project="mvp_2",
        name="mvp_2_final_training",
        save=True,
        val=True,
        plots=True
    )

if __name__ == "__main__":
    main()