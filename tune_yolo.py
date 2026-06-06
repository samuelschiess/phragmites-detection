import os
import sys
import argparse
import torch
from ultralytics import YOLO

CURR_DATASET = "mvp_2"

def main():
    # Verify this will run on GPU if available
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

    # Setup directories
    WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = os.path.join(WORKSPACE_DIR, "datasets", CURR_DATASET, "data.yaml")

    if not os.path.exists(DATA_YAML):
        print(f"[!] Error: data.yaml not found at {DATA_YAML}")
        print("You must have a data.yaml file in the data directory. Add this file or update the path to your data")
        sys.exit(1)

    print(f"[*] Found data.yaml at {DATA_YAML}")

    # Load model
    model_name = "yolo26n-obb.pt"
    print(f"[*] Loading pre-trained model: {model_name}")
    model = YOLO(model_name)
    print(f"[*] Model loaded successfully: {model_name}")

    # Tune the hyperparameters 
    print("[*] Starting hyperparameter tuning...")
    model.tune(
        data=DATA_YAML,
        epochs=30,
        iterations=50,
        optimizer="AdamW",
        patience=5,                    
        use_ray=True,          
        device=device,
        project=CURR_DATASET,
        name="obb_tuning",
        save=True,
        val=True,
        plots=True,
        gpu_per_trial=1.0
    )   

if __name__ == "__main__":
    main()