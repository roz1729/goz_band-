import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Modeli yükle
model = load_model("autoencoder_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# Test için bir görsel al
img_path = "test/add.jpeg"  # test görselini buraya koy
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Tahmin yap (rekonstrüksiyon)
reconstructed = model.predict(img_array)

# Rekonstrüksiyon hatası hesapla
mse = np.mean((img_array - reconstructed) ** 2)
print(f"🔍 Rekonstrüksiyon Hatası (MSE): {mse:.5f}")

# Eşik belirle (örnek: 0.002)
threshold = 0.0015
if mse > threshold:
    print("❌ Anomali tespit edildi!")
else:
    print("✅ Normal örnek.")
