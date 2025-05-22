from PIL import Image
import os

input_folder = "goz_pedleri/normal"
output_folder = "hazir/normal"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpeg") or filename.endswith(".png"):
        img = Image.open(os.path.join(input_folder, filename))
        img = img.resize((224, 224))  # Boyut sabitle
        img.save(os.path.join(output_folder, filename))
