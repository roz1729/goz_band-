# Göz Pedi Kalite Kontrol Sistemi

Bu proje, üretim bandındaki göz pedi ürünlerinin kalite kontrolünü otomatikleştirmek amacıyla geliştirilmiştir. Proje kapsamında, kusursuz (defectsiz) 50 adet göz pedi görseli kullanılarak **anomaly detection (anomali tespiti)** gerçekleştirilmiş, OpenCV ile görsel işleme ve TensorFlow Lite ile model çalıştırma yapılmıştır.

## 🔍 Projenin Amacı

Ped ürünlerinin konum, simetri, sağlamlık ve renk/leke gibi kalite kriterlerini değerlendirerek, hata oranına göre sınıflandırma yapılması ve mobil bir arayüzle bu verilerin kullanıcıya raporlanması amaçlanmaktadır.

## 🧠 Kullanılan Teknolojiler

- **TensorFlow Lite**: Eğitilmiş anomaly tespit modelinin mobil uyumlu çalıştırılması.
- **NumPy**: Görüntü verilerinin matematiksel işlenmesi.
- **OpenCV**: Görsel verilerin işlenmesi ve analiz edilmesi.
- **Python**: Ana programlama dili.
- **Anomaly Detection**: Yalnızca kusursuz örneklerle eğitim yapılarak kusurlu örneklerin tespiti.


