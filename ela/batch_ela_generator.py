import os
from ela_generator import generate_ela_image

# Base project directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input folders
tp_folder = os.path.join(base_dir, "data", "dataset", "Tp")
au_folder = os.path.join(base_dir, "data", "dataset", "Au")

# Output folders
ela_tp_folder = os.path.join(base_dir, "data", "ela_dataset", "Tp")
ela_au_folder = os.path.join(base_dir, "data", "ela_dataset", "Au")

# Create output folders if not exist
os.makedirs(ela_tp_folder, exist_ok=True)
os.makedirs(ela_au_folder, exist_ok=True)

def process_folder(input_folder, output_folder):
    count = 0
    for file in os.listdir(input_folder):
        if file.lower().endswith(".jpg"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            try:
                generate_ela_image(input_path, output_path, quality=90)
                count += 1
            except Exception as e:
                print("❌ Error processing:", file, "->", e)

    print(f"✅ Processed {count} images from {input_folder}")

# Run for both folders
print("Starting ELA generation for Tp (Tampered)...")
process_folder(tp_folder, ela_tp_folder)

print("Starting ELA generation for Au (Authentic)...")
process_folder(au_folder, ela_au_folder)

print("🎉 Batch ELA generation completed!")