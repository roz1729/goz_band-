import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# TFLite modeli yükle
interpreter = tf.lite.Interpreter(model_path="autoencoder_model.tflite")
interpreter.allocate_tensors()

# Girdi / Çıktı detaylarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test görüntüsünü hazırla
img_path = "test/azz.jpeg"
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# Girdiyi ayarla
interpreter.set_tensor(input_details[0]['index'], img_array)

# Tahmin yap
interpreter.invoke()

# Çıktıyı al
reconstructed = interpreter.get_tensor(output_details[0]['index'])

# MSE hesapla
mse = np.mean((img_array - reconstructed) ** 2)
print(f"🔍 Rekonstrüksiyon Hatası (MSE): {mse:.5f}")

# Eşik kontrolü
threshold = 0.0015
if mse > threshold:
    print("❌ Anomali tespit edildi!")
else:
    print("✅ Normal örnek.")
