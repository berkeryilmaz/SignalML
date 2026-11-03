# SignalML — Python Tabanlı Node-Based No-Code / Low-Code Machine Learning Platform

SignalML, modern web tarayıcısı üzerinde çalışan sürükle-bırak **node-based** (Blender Geometry Nodes / Unreal Engine Blueprint tarzı) görsel programlama arayüzüne sahip; analog ve zaman serisi sinyalleri işlemek, derin öğrenme (Keras Functional API) modelleri tasarlamak, eğitmek, ara katman çıktısı çıkarmak ve kümeleme/PCA analizleri yapmak için geliştirilmiş genel amaçlı bir makine öğrenimi platformudur.

---

## 🚀 Önemli Mimari Özellikler

- **%100 Python Backend**: FastAPI, Uvicorn, TensorFlow/Keras, NumPy, Pandas, SciPy, scikit-learn ve SQLite.
- **Sıfır Node.js Bağımlılığı**: Node.js runtime, `npm`, `npx`, `vite` veya `webpack` kesinlikle **kullanılmaz**.
- **İnternetsiz / Offline Çalışma**: LiteGraph, Highcharts ve PapaParse gibi frontend kütüphaneleri `static/vendor` altında yerel olarak sunulur; harici CDN zorunluluğu yoktur.
- **Gerçek Keras Functional API**: Görsel editörde çizilen düğüm grafikleri yalnızca bir çizim olarak kalmaz; backend'de gerçek `keras.Model` nesnelerine dönüştürülür, katman tensör şekilleri (shape inference) hesaplanır ve doğrulanır.
- **Asenkron Eğitim Yönetimi**: Eğitim işleri arka plan iş parçacığında çalışır; HTTP isteklerini bloke etmez, canlı metrik grafikleri (loss, accuracy) anlık olarak takip edilir ve istenildiğinde iptal edilebilir.
- **Güvenlik Koruması**: Güvenli ZIP çıkarma (Zip Slip ve Zip Bomb koruması), kontrollü `/data` klasör erişimi ve güvenli dosya sınırları.
- **Kalıcı Depolama**: Docker volume üzerinden projeler, veri setleri, modeller ve kontrol noktaları SQLite ve diskte kalıcıdır.

---

## 🐳 Docker ile Başlatma

SignalML, Docker ve Docker Compose ile tek komutla ayağa kalkar:

### 1. Varsayılan (CPU Modu)

```bash
cd SignalML
docker compose up --build
```

Uygulama hazır olduğunda tarayıcınızdan şu adrese erişebilirsiniz:
👉 **[http://localhost:8080](http://localhost:8080)**

API Dokümantasyonu (Swagger UI):
👉 **[http://localhost:8080/docs](http://localhost:8080/docs)**

### 2. NVIDIA GPU Desteği ile Başlatma (Opsiyonel)

Eğer makinenizde NVIDIA GPU ve NVIDIA Container Toolkit kuruluysa:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

### 3. Konteyneri Durdurma ve Log İnceleme

```bash
# Arka planda çalıştırma
docker compose up -d

# Canlı logları görüntüleme
docker compose logs -f

# Durdurma
docker compose down
```

### 4. Host Veri Klasörünü Bağlama (`/data`)

Host makinenizdeki veri dosyalarını doğrudan `SignalML/data` klasörüne kopyalayabilir veya `compose.yaml` içerisindeki volume satırını düzenleyebilirsiniz:
```yaml
volumes:
  - ./storage:/app/storage
  - /path/to/my/local/signals:/data:ro
```
Uygulama içindeki **Data Manager &rarr; Mounted Host Directory** bölümünden bu dosyalara kontrollü ve güvenli şekilde erişebilirsiniz.

---

## 📑 Platform Modülleri ve Özellikleri

### 1. Dashboard (`/dashboard`)
Platformun genel durumunu, aktif projeyi, kayıtlı veri seti ve model sayılarını, TensorFlow/GPU çalışma durumunu ve son eğitim geçmişini gösterir.

### 2. Projects (`/projects`)
Birden çok bağımsız çalışma alanı (workspace) oluşturma, projeler arasında geçiş yapma ve proje metadata'sını SQLite üzerinde kalıcı tutma.

### 3. Data Manager (`/data-manager`)
- **Güvenli ZIP Yükleme**: Zip Slip (dizin geçişi) ve Zip Bomb korumalı ZIP açma, içerik listeleme.
- **Mounted `/data` Gezgini**: Host üzerindeki bağlı klasördeki CSV, TXT, TSV, JSON, NPY, NPZ dosyalarını keşfetme.
- **Veri Doğrulama ve Önizleme**: Sinyal boyutları, eksik değerler (missing values), sayısal özet istatistikler (min, max, mean, std).

### 4. Signal Tagger (`/signal-tagger`)
- Yüksek çözünürlüklü etkileşimli dalga formu (Highcharts).
- Sinyaller arasında gezinme (Previous, Next, direkt örnek seçimi).
- **Multi-Signal Overlay**: Birden fazla sinyali aynı grafik üzerinde çakıştırarak karşılaştırma.
- Yakınlaştırma (zoom) ve kaydırma (pan).

### 5. Signal Processing (`/signal-processing`)
Sıralı dönüşüm zinciri (pipeline) kurma ve öncesi/sonrası görsel karşılaştırması:
- **To Fixed Length**: Tüm sinyalleri tam $N$ uzunluğuna getirme (pad/truncate kombinasyonu).
- **First N / Last N**: İlk veya son $N$ örneği seçme.
- **Align Starts / Align Ends**: Belirli enerji veya genlik eşiğine göre sinyal başlangıç ve bitişlerini hizalama.
- **Peak Windowing**: Asimetrik tepe noktası pencereleme ($[Peak - M : Peak + K]$) veya simetrik pencereleme.
- **Slice**: Belirlenen başlangıç, bitiş ve adım (stride) ile dilimleme.
- **Normalize**: Z-Score (ortalama 0, standart sapma 1) ve Min-Max ölçekleme.
- **Missing Value Handling**: Enterpolasyon veya sabit değer doldurma.
- **Exact-Length Export**: CSV, TXT, NPY veya NPZ formatında her satırda tam $N$ örnek içeren, gereksiz indeks sütunu barındırmayan export.

### 6. Visual Workflow Builder (`/workflow-builder`)
Blender / Blueprint tarzı genel amaçlı DAG görsel iş akışı editörü:
- Düğümler: `Data Source` &rarr; `CSV Parser` &rarr; `Preprocessing` &rarr; `Feature Extractor` &rarr; `Clustering` &rarr; `PCA` &rarr; `Export`.
- Tip kontrollü portlar (`Signal`, `Dataset`, `Feature Matrix`, `Model`, `Cluster Results`, `PCA Coordinates`).
- Döngü tespiti (Cycle detection) ve bağımlılık sırasına göre aşamalı çalıştırma.
- Her adım için anlık süre ve çıktı önizlemesi.

### 7. Neural Network Builder (`/nn-builder`)
Gerçek TensorFlow/Keras Functional API modelleri tasarlamak için özel node editörü:
- **Desteklenen Katmanlar**:
  - *Core*: Input, Dense, Activation, Flatten, Reshape, Dropout, BatchNormalization, LayerNormalization.
  - *CNN (1D & 2D)*: Conv1D, SeparableConv1D, Conv2D, MaxPooling1D/2D, AveragePooling1D/2D, GlobalMaxPooling, GlobalAveragePooling.
  - *Recurrent*: LSTM, GRU, SimpleRNN, Bidirectional.
  - *Attention & Transformer*: MultiHeadAttention, PositionalEncoding.
  - *Birleştirme*: Concatenate, Add (Residual bağlantı), Multiply.
- **Otomatik Şekil Çıkarımı (Shape Inference)**: Editördeki katmanların girdi/çıktı tensör boyutlarını ve toplam parametre sayılarını backend doğrular.
- **Keras Modeline Derleme**: Derlenen model `.keras` formatında kaydedilir.

### 8. Training (`/training`)
- Arka planda asenkron eğitim işi (Job ID).
- Adam, SGD, RMSprop, AdamW optimizer desteği.
- Canlı metrik eğrileri (Training Loss, Validation Loss, Accuracy).
- Erken durdurma (Early Stopping) ve kullanıcı tarafından iptal (Stop Training).
- Değerlendirme raporu: Confusion Matrix ve sınıflandırma metrikleri.

### 9. Feature Extraction (`/feature-extraction`)
- Eğitilmiş Keras modelindeki herhangi bir gizli katmanı (örn. LSTM veya Conv1D) seçme.
- Keras alt modeli (`keras.Model(inputs=model.inputs, outputs=selected_layer.output)`) oluşturma.
- Tüm veri seti üzerinde toplu (batch) tensör çıktısı alma ve öznitelik matrisi olarak saklama.

### 10. Clustering & PCA (`/clustering`)
- **Algoritmalar**: K-Means ($k$, $n\_init$) ve DBSCAN ($eps$, $min\_samples$).
- **Veri Kaynakları**: Ham sinyaller veya sinir ağından çıkarılan ara katman öznitelikleri.
- **Etkileşimli 3D PCA**: Fare ile döndürülebilen, yakınlaştırılabilen 3D uzayda küme renklendirmeli saçılım grafiği, aykırı değer (outlier) tespiti ve örnek bazlı tooltip.

### 11. Model Evaluation (`/model-evaluation`)
Model performans karnesi, geçmiş eğitim deneylerinin karşılaştırmalı tablosu.

### 12. Exports (`/exports`)
Eğitilen Keras model dosyalarını (`.keras`), tam $N$ uzunluktaki sinyal dosyalarını ve öznitelik matrislerini tek tıkla indirme.

---

## 🧪 Pytest Testleri

Platformun tüm bileşenleri için kapsamlı 28 birim ve entegrasyon testi hazırlanmıştır:

```bash
# Testleri çalıştırmak için:
python3 -m pytest backend/tests -v
```

Test Kapsamı:
1. `test_csv_and_txt_loading`: CSV/TXT yükleme ve satır bazlı sinyal ayrıştırma.
2. `test_npy_npz_loading`: NumPy binary yükleme.
3. `test_zip_safety_and_extraction`: Güvenli ZIP ve Zip Slip koruması.
4. `test_first_n_and_last_n`: İlk N ve son N örnek seçimi.
5. `test_pad_and_truncate`: Sabit, kenar ve yansıtmalı padding ile kırpma.
6. `test_align_starts_and_ends`: Eşik tabanlı sinyal başlangıç/bitiş hizalama.
7. `test_peak_windowing`: Asimetrik tepe noktası pencereleme.
8. `test_exact_n_signals_export`: Tam N örnekli, indekssiz export doğrulama.
9. `test_node_graph.py`: Node graph doğrulaması, port tipleri ve döngü tespiti.
10. `test_keras_builder.py`: Gerçek Keras Functional API derlemesi ve tensör şekilleri.
11. `test_training.py`: Gerçek veri üzerinde Keras model eğitimi ve metrik doğrulaması.
12. `test_feature_extraction.py`: Katman ara çıktılarının alt modelle çıkarılması.
13. `test_clustering_pca.py`: K-Means, DBSCAN, 2D ve 3D PCA hesaplamaları.
14. `test_workflow_engine.py`: Uçtan uca DAG akışının işletilmesi ve kalıcı depolama.
15. `test_api_integration.py`: FastAPI endpoint'leri ve 12 sayfa render testi.

---

## 📂 Dizin Yapısı

```text
SignalML/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI REST router'ları
│   │   ├── core/             # Konfigürasyon, SQLite veritabanı, güvenlik
│   │   ├── schemas/          # Pydantic veri modelleri
│   │   ├── services/         # Sinyal işleme, Keras derleyici, DAG motoru vb.
│   │   ├── templates/        # 12 adet Jinja2 HTML sayfası
│   │   ├── static/
│   │   │   ├── css/          # Blueprint/Dark mod tasarım CSS dosyaları
│   │   │   ├── js/           # Modül kontrolcüleri ve visualizer JS dosyaları
│   │   │   └── vendor/       # Yerel kütüphaneler (litegraph, highcharts, papaparse)
│   │   └── main.py           # FastAPI başlangıç dosyası
│   ├── tests/                # 28 adet Pytest testi
│   └── requirements.txt      # Sabitlenmiş Python bağımlılıkları
├── data/                     # Host veri paylaşım klasörü (/data)
├── storage/                  # Kalıcı Docker depolama alanı
├── Dockerfile                # Python 3.12 tabanlı tek container imajı
├── compose.yaml              # Docker Compose konfigürasyonu
├── compose.gpu.yaml          # NVIDIA GPU override dosyası
├── .dockerignore
├── .env.example
└── README.md
```

---

## 🔒 Güvenlik

- **Path Traversal Koruması**: `resolve_safe_path` ve `extract_zip_safely` ile dizin dışına çıkma girişimleri engellenir.
- **Zip Bomb Koruması**: Arşiv içi maksimum dosya sayısı ve toplam açılmış boyut sınırlandırılmıştır.
- **Arbitrary Kod Çalıştırma Engeli**: Frontend'den gelen dinamik kodlar çalıştırılmaz; tüm işlemler backend'deki doğrulanmış Python servisleri üzerinden yönetilir.
