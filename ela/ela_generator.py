from PIL import Image, ImageChops, ImageEnhance
import os

def generate_ela_image(image_path, output_path, quality=90):
    # Open original image
    original = Image.open(image_path).convert("RGB")

    # Save temporary compressed image
    temp_path = "temp_ela.jpg"
    original.save(temp_path, "JPEG", quality=quality)

    # Open compressed image
    compressed = Image.open(temp_path)

    # Compute difference
    ela_image = ImageChops.difference(original, compressed)

    # Normalize brightness
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # Save final ELA image
    ela_image.save(output_path)

    # Remove temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print("✅ ELA image saved at:", output_path)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    test_image = os.path.join(
        base_dir,
        "data",
        "dataset",
        "Tp",
        "Tp_S_NRN_S_N_sec00065_sec00065_11280.jpg"
    )

    output_image = os.path.join(base_dir, "ela", "ela_result.jpg")

    generate_ela_image(test_image, output_image, quality=90)