# Laporan Proyek Machine Learning - Indira Aline


## Domain Proyek

Domain yang dipilih untuk proyek *machine learning* ini adalah **Medical**, dengan judul **Predictive Analytics: Diabetes Prediction**  


## Latar Belakang Proyek Prediksi Diabetes Menggunakan Model GAN

![Foto Diabetes](https://static.banyumaskab.go.id/website/images/website_1111201037225fab5c72eb04f.jpg)

Diabetes melitus adalah salah satu penyakit kronis yang menjadi masalah kesehatan global, dengan prevalensi yang terus meningkat setiap tahunnya. Menurut International Diabetes Federation (IDF), sekitar 537 juta orang dewasa (20-79 tahun) hidup dengan diabetes pada tahun 2021, dan angka ini diproyeksikan meningkat menjadi 643 juta pada tahun 2030. [[1](https://diabetesatlas.org/)] Deteksi dini diabetes sangat penting untuk mencegah komplikasi serius seperti penyakit jantung, gagal ginjal, dan kerusakan saraf. [[2](https://diabetesjournals.org/care/article/45/Supplement_1/S1/138921/Introduction-Standards-of-Medical-Care-in-Diabetes)]

Dalam upaya meningkatkan akurasi prediksi diabetes, berbagai model machine learning telah dikembangkan, seperti Logistic Regression, Random Forest, dan Support Vector Machines (SVM). Namun, salah satu tantangan utama dalam pengembangan model prediksi diabetes adalah ketidakseimbangan data (class imbalance), di mana jumlah pasien non-diabetes jauh lebih banyak daripada pasien diabetes. [[3](https://arxiv.org/abs/1406.2661)] Hal ini dapat menyebabkan model cenderung memprediksi kelas mayoritas dan mengabaikan minoritas, sehingga mengurangi performa prediksi.

Generative Adversarial Networks (GAN) merupakan salah satu pendekatan deep learning yang dapat digunakan untuk mengatasi masalah ketidakseimbangan data dengan menghasilkan sampel sintetis yang mirip dengan data asli. [[4](https://arxiv.org/abs/1803.09655)] Dengan memanfaatkan GAN, dapat menyeimbangkan dataset sebelum melatih model klasifikasi, sehingga meningkatkan akurasi prediksi diabetes.

Proyek ini bertujuan untuk mengembangkan model prediksi diabetes yang lebih akurat dengan memanfaatkan GAN untuk mengatasi masalah ketidakseimbangan data dan meningkatkan kemampuan prediktif model.


## Business Understanding

✅ Bagi Tenaga Medis: Membantu diagnosis dini dengan akurasi lebih tinggi.
✅ Bagi Pasien: Deteksi risiko diabetes lebih awal untuk pencegahan yang lebih baik.
✅ Bagi Peneliti Kesehatan: Dataset sintetis dapat digunakan untuk eksperimen tanpa melanggar privasi pasien.

### Problem Statements

Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- **Bagaimana cara meningkatkan akurasi prediksi diabetes dengan data yang tidak seimbang?**
- **Dapatkah Generative Adversarial Networks (GAN) menghasilkan data sintetis yang realistis untuk memperbaiki model prediksi?**
- **Bagaimana menangkap hubungan non-linier dan interaksi kompleks antara berbagai faktor risiko diabetes (seperti BMI, glukosa darah, usia) untuk prediksi yang lebih akurat?**

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- **Mengatasi Data Tidak Seimbang**: Menggunakan GAN untuk menghasilkan data sintetis pasien diabetes guna menyeimbangkan dataset.
- **Meningkatkan Akurasi Prediksi**: Membangun model prediksi (seperti Neural Network atau XGBoost) yang dilatih pada data asli + data sintetis dari GAN.
- **Dengan arsitektur adversarial**, GAN dapat belajar distribusi data yang lebih baik dan menangkap pola tersembunyi dalam faktor risiko diabetes.

    ### Solution statements
    1.  Membangun GAN untuk Data Sintetis Diabetes
        -   Generator akan mempelajari distribusi data pasien diabetes asli dan menghasilkan sampel sintetis yang realistis
        -   Discriminator akan membantu meningkatkan kualitas data yang dihasilkan agar tidak mudah dibedakan dari data nyata.

    2.  Menggunakan Data Sintetis untuk Pelatihan Model Prediksi:
        - Setelah dataset seimbang, model klasifikasi (seperti Neural Network) akan dilatih untuk memprediksi diabetes dengan lebih akurat.


## Data Understanding

### Deskripsi Variabel
**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | _Diabetes prediction dataset_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?resource=download) |
| Maintainer | [Mohammed Mustafa ⚡](https://www.kaggle.com/iammustafatz) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | _Healty, Diabetes, Classification, Healthcare, Binary Classification_ |
| Usability | 10.00 |

Berikut informasi pada dataset: 
Data yang digunakan dalam pembuatan model merupakan data primer, data ini adalah kumpulan data medis dan demografis dari pasien, yang disediakan secara publik di Kaggle dengan nama datasets yaitu: _Diabetes Prediction Dataset_

| No. | Gender | Age | Hypertension | Heart Disease | Smoking History | BMI | HbA1c_level | Blood Glucose Level | Diabetes |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 0 | Female | 80.00 | 0 | 1 | Never | 25.19 | 6.6  | 140 | 0 |
| 1 | Female | 54.00 | 0 | 0 | No info | 27.32 | 6.6  | 80 | 0 |
| 2 | Male | 28.00 | 0 | 0 | Never | 27.32	| 5.7 | 158 | 0 |
| 3 | Female | 36.00 | 0 | 0 | Current | 23.45	| 5.0  | 155 | 0 |
| 4 | Male |76.00 | 1 | 1 | Current | 20.14	| 4.8  | 155 | 0 |

Tabel 1. Data Loading Variabel

- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 100.000 sample dengan 9 fitur.
- Dataset memiliki 3 fitur bertipe float64 , 2 fitur bertipe object dan 4 fitur bertipe int64

### Variable-Variable Pada Dataset

- `Gender` : Jenis kelamin pasien.
- `Age` : Usia pasien dalam tahun.
- `Hypertension` : Apakah pasien memiliki hipertensi (1 = Ya, 0 = Tidak).
- `Heart Disease` : Apakah pasien memiliki penyakit jantung (1 = Ya, 0 = Tidak).
- `Smoking History` : Riwayat merokok pasien.
- `BMI` : Indeks Massa Tubuh.
- `HbA1c_level` : Kadar hemoglobin A1c (rata-rata glukosa darah dalam 3 bulan).
- `Blood Glucose Level` : Kadar glukosa darah acak (mg/dL).
- `Diabetes` : Target variabel (1 = Diabetes, 0 = Tidak).

### Latih Data

![Training Loss 1](/image/TrainingLoss1.png) 

Gambar 1a.  Training Loss 1

-   Rentang epoch lebih pendek (0-200)
-   Skala loss lebih rendah (0.25-1.75)
-   Mungkin menunjukkan konvergensi yang lebih baik atau tahap pelatihan yang berbeda

![Training Loss 2](/image/TrainingLoss2.png)

Gambar 1b.  Traininng Loss 2

-   Rentang epoch lebih panjang (0-400)
-   Skala loss lebih tinggi (0.0-3.0)
-   Menunjukkan pelatihan dalam jangka panjang dengan fluktuasi loss yang lebih besar


## Data Preparation

Teknik Data Preparation yang Dilakukan:

a. Encoding Data Kategorikal
    -   Kolom 'gender' diubah dari nilai kategorikal ('Female', 'Male', 'Other') menjadi numerik (0, 1, 2)
    -   Kolom 'smoking_history' diubah dari teks ('No Info', 'never', dll) menjadi numerik (0-5)

Alasan:
-   Algoritma machine learning umumnya hanya bisa memproses data numerik
-   Mengubah data kategorikal menjadi numerik memungkinkan model untuk memproses informasi ini
-   Pemetaan nilai yang konsisten penting untuk interpretasi hasil

b. Pembagian Fitur dan Label
    Memisahkan dataset menjadi:
    -   Fitur (X): ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbAic_level', 'blood_glucose_level']
    -   Label (y): ['diabetes']

Alasan:
-   Memisahkan variabel independen (fitur) dan dependen (label) adalah langkah dasar dalam supervised learning
-   Membantu dalam proses pelatihan model dan evaluasi performa
-   Memastikan model belajar hubungan antara fitur dan target

c. Pembagian Data Latih dan Uji
    Menggunakan train_test_split dengan:
    -   test_size=0.10 (10% data untuk testing)
    -   random_state=0 (untuk reproduktibilitas)

Alasan:
-   Mengevaluasi kemampuan generalisasi model pada data yang belum pernah dilihat
-   Mencegah overfitting dengan memisahkan data evaluasi dari data pelatihan
-   Rasio 90:10 menunjukkan fokus pada pelatihan model dengan tetap memiliki cukup data untuk evaluasi

d. Pemodelan Awal dengan Random Forest
    -   Membuat model RandomForestClassifier dengan 10 estimator
    -   Melatih model dan mengevaluasi performa dasar

Alasan:
-   Memberikan baseline performance untuk perbandingan dengan model yang lebih kompleks
-   Random Forest dipilih karena kemampuannya menangani berbagai jenis fitur tanpa perlu normalisasi ekstensif
-   Jumlah estimator yang kecil (10) untuk komputasi cepat dalam tahap eksplorasi

## Modeling

1. **Pembuatan Generative Adversarial Network (GAN)**
    a.  **Generator**:
        -   Arsitektur: 3 layer Dense (15, 30, dan 9 unit)
        -   Fungsi aktivasi: ReLU untuk layer hidden, linear untuk output
        -   Inisialisasi bobot: He uniform
        -   Input: Vektor laten berdimensi 10
        -   Output: 9 fitur sintetik (sesuai dengan fitur dataset asli)

    b.  **Discriminator**:
        -   Arsitektur: 3 layer Dense (25, 50, dan 1 unit)
        -   Fungsi aktivasi: ReLU untuk layer hidden, sigmoid untuk output
        -   Loss function: Binary crossentropy
        -   Optimizer: Adam

    c.  **Proses Pelatihan GAN**:
        -   Batch size: 128 (dibagi 64 sampel nyata dan 64 sintetik)
        -   Epoch: 500
        -   Evaluasi setiap 200 epoch
        -   Pelatihan adversarial dimana generator dan discriminator saling bersaing

2. **Pembuatan Model Klasifikasi dengan Random Forest**
    a.  **Parameter**
        -   n_estimators: 10
        -   Data dibagi 70% training dan 30% testing
        -   random_state: 42 untuk reproduktibilitas


**Kelebihan dan Kekurangan Algoritma**

**GAN (Generator)**

Kelebihan:
-   Dapat menghasilkan data sintetik yang mirip dengan distribusi data asli
-   Berguna ketika data asli terbatas atau tidak seimbang
-   Mampu menangkap hubungan kompleks dalam data

Kekurangan:
-   Proses pelatihan tidak stabil dan sulit dikonvergensi
-   Membutuhkan banyak komputasi
-   Hasil sulit diinterpretasi (output berupa bilangan kontinu)

**Random Forest**

Kelebihan:
-   Robust terhadap overfitting
-   Dapat menangani berbagai jenis fitur (numerik/kategorik)
-   Memberikan feature importance
-   Performa bagus out-of-the-box

Kekurangan:
-   Kurang performa pada data sangat tinggi dimensi
-   Model cenderung besar (banyak pohon)
-   Kurang interpretabel dibanding single decision tree

**Pemilihan Model Terbaik**
Berdasarkan hasil:
a.  Model dengan data asli: Akurasi 97% (tapi recall kelas minoritas 70%)
b.  Model dengan data sintetik: Akurasi 89.7% dengan recall lebih seimbang


## Evaluation

Metrik Evaluasi yang Digunakan:

1. **Statistik Deskriptif dan Uji KS (Kolmogorov-Smirnov)**
Formula:

![Kolmogorov-Smirnov](/image/Kolmogorov-Smirnov.png)

Penjelasan:
-   Mean dan std mengukur pusat dan sebaran distribusi data
-   Uji KS membandingkan distribusi kumulatif dua sampel
-   p-value < 0.05 menunjukkan perbedaan signifikan

2. **Visualisasi PCA (Principal Component Analysis)**
Formula:

![Principal Component Analysis](/image/PCA.png)

Penjelasan:
-   Mereduksi dimensi data menjadi 2 komponen utama
-   Memvisualisasikan kemiripan distribusi data asli dan sintetik

3. **Metrik Klasifikasi (Untuk Model Random Forest)**
Formula:

![Metrik Klasifikasi](/image/Metrik-Klasifikasi.png)

Penjelasan:
-   Mengukur performa model klasifikasi pada data sintetik
-   F1-score khusus penting untuk data tidak seimbang

![PCA Comparison: Real vs Synthetic Data](/image/Real_vs_Synthetic_Data.png)

### Hasil Evaluasi

**1. Kualitas Data Sintetik**

Hasil Statistik:
-   Semua kolom menunjukkan perbedaan mean dan std yang signifikan
-   Nilai p-value KS test = 0 untuk hampir semua kolom, menunjukkan distribusi sangat berbeda
-   Contoh perbedaan ekstrim:
    -   Age: Asli (μ=41.9) vs Sintetik (μ=-4.2)
    -   Blood glucose: Asli (μ=138.1) vs Sintetik (μ=0.24)
    -   Diabetes: Asli (8.5% positif) vs Sintetik (53.1% positif)

Analisis PCA:
-   Plot menunjukkan overlap minimal antara cluster data asli dan sintetik
-   Data sintetik memiliki sebaran yang berbeda di ruang PCA



## Referensi

1. International Diabetes Federation. (2021). IDF Diabetes Atlas, 10th edn. Brussels, Belgium. https://diabetesatlas.org/
2. American Diabetes Association. (2022). Standards of Medical Care in Diabetes. Diabetes Care, 45(Supplement_1). https://doi.org/10.2337/dc22-SINT
3. Goodfellow, I., et al. (2014). Generative Adversarial Networks. arXiv:1406.2661. https://arxiv.org/abs/1406.2661
4. Mariani, G., et al. (2018). BAGAN: Data Augmentation with Balancing GAN. arXiv:1803.09655. https://arxiv.org/abs/1803.09655
