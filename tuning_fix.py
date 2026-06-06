import os
import json
import yaml

def rescue_best_hyperparameters():
    # The exact path you located
    tune_dir = r"C:\Users\schie\Desktop\Phragmites\runs\detect\mvp_2\obb_tuning"
    
    best_fitness = 0.0
    best_params = {}
    best_trial = ""

    print("[*] Scanning Ray Tune trial folders...")

    for trial_name in os.listdir(tune_dir):
        trial_path = os.path.join(tune_dir, trial_name)
        
        # Only look at the actual trial directories
        if os.path.isdir(trial_path) and trial_name.startswith("_tune"):
            result_file = os.path.join(trial_path, "result.json")
            params_file = os.path.join(trial_path, "params.json")

            if os.path.exists(result_file) and os.path.exists(params_file):
                try:
                    # Ray Tune saves results line-by-line per epoch. We want the last line.
                    with open(result_file, 'r') as f:
                        lines = f.readlines()
                        if not lines:
                            continue
                        last_result = json.loads(lines[-1])
                    
                    # YOLO calculates fitness as: 10% mAP50 + 90% mAP50-95
                    map50 = last_result.get("metrics/mAP50(B)", 0)
                    map50_95 = last_result.get("metrics/mAP50-95(B)", 0)
                    fitness = (0.1 * map50) + (0.9 * map50_95)

                    # Update if this is the new highest score
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_trial = trial_name
                        with open(params_file, 'r') as pf:
                            best_params = json.load(pf)
                            
                except Exception as e:
                    print(f"[-] Skipped {trial_name} due to read error.")
                    continue

    if not best_params:
        print("[!] Could not find any valid results. Check your directory path.")
        return

    print(f"\n[+] Rescue Complete!")
    print(f"    Winning Trial: {best_trial}")
    print(f"    Fitness Score: {best_fitness:.5f}")

    # Save it identically to how Ultralytics would have
    output_path = "best_hyperparameters.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
        
    print(f"[+] Successfully generated: {output_path}")

if __name__ == "__main__":
    rescue_best_hyperparameters()