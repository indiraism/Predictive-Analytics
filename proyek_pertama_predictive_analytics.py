# -*- coding: utf-8 -*-
"""Proyek Pertama : Predictive Analytics.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15tu2OjYd1x3roQbBZ8x00t_IKbxekbmM

# **Predictive Analytics Project: [(Diabetes Prediction)](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?resource=download)**
- **Nama:** Indira Aline
- **Email:** indiradira63@gmail.com
- **ID Dicoding:** indira_kbs

## **About Dataset**

The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans. Additionally, the dataset can be used by researchers to explore the relationships between various medical and demographic factors and the likelihood of developing diabetes.

# **Import Semua Packages/Library yang Digunakan**
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from numpy.random import randn
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA

from scipy import stats

# Mount Google Drive (Colab)
from google.colab import files
from google.colab import drive
drive.mount('/content/drive')

"""# **Data Loading**

Dataset yang akan digunakan bernama diabetes_prediction_dataset.csv
"""

# Import module yang disediakan google colab untuk kebutuhan upload file

from google.colab import files
files.upload()

# Setup Kaggle API authentication
os.makedirs('/root/.kaggle', exist_ok=True)
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!ls -l /root/.kaggle/

!kaggle datasets download -d iammustafatz/diabetes-prediction-dataset

import zipfile,os,shutil

fileZip = "diabetes-prediction-dataset.zip"
extracZip = zipfile.ZipFile(fileZip, 'r')
extracZip.extractall("dataset")

os.listdir("/content/dataset")

data = pd.read_csv('dataset/diabetes_prediction_dataset.csv')

"""About this file

The diabetes_prediction_dataset.csv file contains medical and demographic data of patients along with their diabetes status, whether positive or negative. It consists of various features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. The Dataset can be utilized to construct machine learning models that can predict the likelihood of diabetes in patients based on their medical history and demographic details.

# **Exploratory Data Analysis**

## Struktur Data
"""

# Menampilkan informasi tentang dimensi data
print(data.shape)

# Menampilakn beberapa baris terakhir dari data
print(data.tail())

# Menampilakn nama-nama kolom pada data
print(data.columns)

# Menampilkan beberapa baris pertama dari data
data.head(10)

data.info()

"""## Missing Values"""

print(f"Jumlah data hilang pada dataset: \n{data.isna().sum()}")

"""## Univariate Analysis"""

## A. Categorical Variables

cat_vars = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']
plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_vars, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=data, x=col, palette='viridis')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

## B. Numerical Variables

num_vars = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
plt.figure(figsize=(15, 8))
for i, col in enumerate(num_vars, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=data, x=col, kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

"""## Bivariate Analysis"""

##A. Diabetes vs. Key Features

plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='diabetes', y='HbA1c_level', palette='pastel')
plt.title("HbA1c Levels in Diabetic vs Non-Diabetic Patients")
plt.show()

## B. Correlation Matrix

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

## Outlier Detection

plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']], palette='Set3')
plt.title("Outlier Detection in Numerical Variables")
plt.show()

## Class Imbalance Analysis

print("\nDiabetes Class Distribution:\n", data['diabetes'].value_counts(normalize=True))

"""# **Data Preprocessing**

#### Split Dataset
"""

# Pra-pemrosesan data
# Mengganti nilai kategorikal pada kolom 'gender' dengan nilai numerik
data['gender'] = data['gender'].replace({'Female': 0, 'Male': 1, 'Other': 2})

# Mengganti nilai kategorikal pada kolom 'smoking_history' dengan nilai numerik
data['smoking_history'] = data['smoking_history'].replace({
    'No Info': 0,
    'never': 1,
    'former': 2,
    'current': 3,
    'not current': 4,
    'ever': 5
})

"""### Memisahkan data menjadi fitur (features) dan label

"""

# Fitur (features) yang akan digunakan dalam pemoodelan
features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Kolom target (label) yang diprediksi
label = ['diabetes']

# Memisahkan data menjadi fitur (X) dan label (y)
X = data[features]
y = data [label]

"""### Membagi data latih dan data uji"""

# Memisahkan data menjadi data latih dan data uji
X_true_train, X_true_test, y_true_train, y_true_test = train_test_split(X, y, test_size=0.10, random_state=0)

# Membuat objek RandomForestClassifier
clf_true = RandomForestClassifier(n_estimators=10)

# Melatih model dengan data latih
clf_true.fit(X_true_train, y_true_train)

# Melakukan prediksi pada data uji
y_true_pred = clf_true.predict(X_true_test)

# Menghitung akurasi prediksi
print("Akurasi Dasar:", metrics.accuracy_score(y_true_test, y_true_pred))

# Menampilkan laporan klasifikasi
print("Laporan klasifikasi dasar:\n", metrics.classification_report(y_true_test, y_true_pred))

"""1.   Akurasi Keseluruhan: 0.9784 (97.84%)
  *   Model ini sangat akurat dalam memprediksi secara keseluruhan


2.   Laporan Klasifikasi:
  *   Kelas 0 (mayoritas):
      *   Precision: 97% - Dari semua prediksi kelas 0, 97% benar
      *   Recall: 100% - Model berhasil mengidentifikasi semua instance kelas 0 yang sebenarnya
      *   F1-score: 0.98 - Keseimbangan antara precision dan recall yang sangat baik

  *   Kelas 1 (minoritas):
      *   Precision: 94% - Dari semua prediksi kelas 1, 94% benar
      *   Recall: 70% - Model hanya berhasil mengidentifikasi 70% dari instance kelas 1 yang sebenarnya
      *   F1-score: 0.80 - Lebih rendah dibanding kelas 0


3.   Ketidakseimbangan Kelas:
  *   Jumlah data: 9141 untuk kelas 0 vs 859 untuk kelas 1
  *   Model lebih baik memprediksi kelas mayoritas (0) dibanding kelas minoritas (1)

## Modelling

Model yang digunakan pada proyek ini adalah **Model GAN (Generative Adversarial Network)**.

**GAN (Generative Adversarial Network)** adalah arsitektur deep learning yang terdiri dari dua jaringan neural yang saling bersaing:

Komponen Utama GAN:

1.   Generator:
  *   Bertugas membuat data sintetis yang menyerupai data asli
  *   Menerima input noise vector dan menghasilkan sampel palsu


2.   Discriminator:
  *   Bertugas membedakan antara data asli dan data palsu dari generator
  *   Bertindak sebagai classifier biner (asli vs palsu)
"""

def generate_latent_points(latent_dim, n_samples):
    # Menghasilkan titik-titik laten acak
    x_input = randn(latent_dim * n_samples)
    # Mengubah bentuk titik-titik laten menjadi matriks
    x_input = x_input.reshape(n_samples, latent_dim)
    # Mengembalikan titik-titik laten yang dihasilkan
    return x_input

# Menggunakan generator untuk menghasilkan m contoh palsu, dengan label kelas
def generate_fake_samples(generator, latent_dim, n_samples):
    # Menghasilkan titik-titik laten acak sebagai input generator
    x_input = generate_latent_points(latent_dim, n_samples)
    # Menghasilkan contoh palsu menggunakan generator
    X = generator.predict(x_input)
    # Membuat array label kelas palsu dengan nilai 0
    y = np.zeros((n_samples, 1))
    # Mengembalikan contoh palsu dan labelnya
    return X, y

# Menghasilkan n contoh nyata dengan label kelas; secara acak memilih n sampel dari data nyata
def generate_real_samples(n):
    # Memilih secara acak n sampel dari data nyata
    X = data.sample(n)
    # Membuat array label kelas nyata dengan nilai 1
    y = np.ones((n, 1))
    return X, y

# Mendefinisikan generator dengan dimensi laten yang diberikan dan jumlah output yang ditentukan (default = 9)
def define_generator(latent_dim, n_outputs=9):
    # Membuat model Sequential
    model = Sequential()
    # Menambahkan layer Dense dengan 15 unit, fungsi aktivasi ReLU, dan inisialisasi bobot He
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    # Menambahkan layer Dense dengan 30 unit dan fungsi aktivasi ReLU
    model.add(Dense(30, activation='relu'))
    # Menambahkan layer Dense dengan jumlah output yang ditentukan dan fungsi aktivasi linear
    model.add(Dense(n_outputs, activation='linear'))
    return model

# Mendefinisikan generator1 dengan dimensi laten 10 dan jumlah output 9
generator1 = define_generator(10, 9)

# Menampilkan ringkasan (summary) dari generator1
generator1.summary()

# Mendefinisikan discriminator dengan jumlah input yang ditentukan (default = 9)
def define_discriminator(n_inputs=9):
    # Membuat model Sequential
    model = Sequential()
    # Menambahkan layer Dense dengan 25 unit, fungsi aktivasi ReLU, dan inisialisasi bobot He
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    # Menambahkan layer Dense dengan 50 unit dan fungsi aktivasi ReLU
    model.add(Dense(50, activation='relu'))
    # Menambahkan layer Dense dengan 1 unit dan fungsi aktivasi sigmoid
    model.add(Dense(1, activation='sigmoid'))
    # Mengompilasi model dengan loss function binary_crossentropy, optimizer adam, dan metrik akurasi
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Mendefinisikan model GAN yang terdiri dari generator dan discriminator, yang digunakan untuk memperbarui generator
def define_gan(generator, discriminator):
    # Membuat bobot pada discriminator tidak dapat di-train
    discriminator.trainable = False
    # Membuat model Sequential
    model = Sequential()
    # Menambahkan generator ke dalam model
    model.add(generator)
    # Menambahkan discriminator ke dalam model
    model.add(discriminator)
    # Mengompilasi model dengan loss function binary_crossentropy dan optimizer adam
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# Membuat line plot dari loss untuk GAN dan menyimpan ke file
def plot_history(d_hist, g_hist):
    # Menampilkan plot loss
    plt.subplot(1, 1, 1)
    plt.plot(d_hist, label='Discriminator')
    plt.plot(g_hist, label='Generator')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.show()
    plt.close()

# Melatih generator dan discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=500, n_batch=128, n_eval=200):
    # Menentukan setengah ukuran satu batch, untuk memperbarui discriminator
    half_batch = int(n_batch / 2)
    d_history = []
    g_history = []

    # Melakukan iterasi secara manual pada setiap epoch
    for epoch in range(n_epochs):
        # Menyiapkan sampel-sampel asli
        x_real, y_real = generate_real_samples(half_batch)
        # Menyiapkan contoh palsu
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # Memperbarui discriminator
        d_loss_real, d_real_acc = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, d_fake_acc = d_model.train_on_batch(x_fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # Menyiapkan titik-titik di dalam ruang laten sebagai input untuk generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # Membuat label terbalik untuk sampel-sampel palsu
        y_gan = np.ones((n_batch, 1))
        # Memperbarui generator melalui kesalahan discriminator
        g_loss_fake = gan_model.train_on_batch(x_gan, y_gan)

        print('%d, d1=%.3f, d2=%.3f, d=%.3f, g=%.3f' % (epoch+1, d_loss_real, d_loss_fake, d_loss, g_loss_fake))

        # Menyimpan nilai loss
        d_history.append(d_loss)
        g_history.append(g_loss_fake)

        # Mengevaluasi model
        if (epoch+1) % n_eval == 0:
            plot_history(d_history, g_history)

    # Menyimpan model generator yang telah dilatih
    g_model.save('trained_generated_model.h5')

# Ukuran dari ruang laten
latent_dim = 10

# Membuat discriminator
discriminator = define_discriminator()

# Membuat generator
generator = define_generator(latent_dim)

# Membuat model GAN
gan_model = define_gan(generator, discriminator)

# Melatih model
train(generator, discriminator, gan_model, latent_dim)

# Memuat model yang telah dilatih
model = load_model('/content/trained_generated_model.h5')

# Menghasilkan titik-titik dalam ruang laten
latent_points = generate_latent_points(10, 750)

# Menggunakan model generator untuk menghasilkan data palsu
X = model.predict(latent_points)

# Membuat DataFrame dari data palsu
data_fake = pd.DataFrame(data=X, columns=['gender', 'age', 'hypertension',
                                         'heart_disease', 'smoking_history', 'bmi',
                                         'HbA1c_level', 'blood_glucose_level', 'diabetes'])

# Menampilkan 5 baris pertama dari data palsu
data_fake.head()

# Menghitung rata-rata diabetes pada data palsu
diabetes_mean = data_fake.diabetes.mean()

# Mengubah nilai diabetes menjadi True jika lebih besar dari rata-rata, dan False jika sebaliknya
data_fake['diabetes'] = data_fake['diabetes'] > diabetes_mean

# Mengubah tipe data diabetes menjadi integer
data_fake["diabetes"] = data_fake["diabetes"].astype(int)

# Menentukan fitur-fitur yang digunakan
features = ['gender', 'age', 'hypertension', 'heart_disease',
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Menentukan label yang digunakan
label = ['diabetes']

# Mengambil fitur-fitur dari data palsu
X_fake_created = data_fake[features]

# Mengambil label dari data palsu
y_fake_created = data_fake[label]

# Membagi data palsu menjadi data latih dan data uji
X_fake_train, X_fake_test, y_fake_train, y_fake_test = train_test_split(
    X_fake_created, y_fake_created, test_size=0.30, random_state=42
)

# Membuat dan melatih model dengan data palsu
clf_fake = RandomForestClassifier(n_estimators=10)
clf_fake.fit(X_fake_train, y_fake_train)

# Memprediksi label menggunakan model data palsu
y_fake_pred = clf_fake.predict(X_fake_test)

# Menghitung akurasi model data palsu
accuracy_fake = metrics.accuracy_score(y_fake_test, y_fake_pred)

# Menampilkan laporan klasifikasi model data palsu
classification_report_fake = metrics.classification_report(y_fake_test, y_fake_pred)

print("Accuracy of fake data model:", accuracy_fake)
print("Classification report of fake data model:")
print(classification_report_fake)

"""*   Model ini memiliki performa yang cukup baik secara keseluruhan, mampu memprediksi dengan benar hampir 90% dari total data.
*   Kelas 1 memiliki performa sedikit lebih baik dibanding Kelas 0
*   Tidak ada ketimpangan performa yang signifikan antara kedua kelas
*   Performa yang seimbang antara kedua kelas
*   Recall dan precision yang relatif seragam untuk masing-masing kelas
*   Tidak ada indikasi bias yang kuat terhadap salah satu kelas

## Evaluasi dan Visualisasi

Tahapan Analisis yang Dilakukan:

1.   Iterasi melalui setiap kolom data:
  *   Kode melakukan loop untuk setiap kolom dalam dataset
  *   Untuk setiap kolom, dibandingkan statistik antara data asli dan sintetik


2.   Perhitungan statistik deskriptif:
  *   Menghitung mean (rata-rata) dan standar deviasi (std) untuk:
      *   Data asli (data[col])
      *   Data sintetik (data_fake[col])


3.   Uji Kolmogorov-Smirnov (KS Test):
  *   Dilakukan untuk menguji apakah distribusi data sintetik sama dengan data asli
  *   Menghasilkan p-value yang menunjukkan signifikansi perbedaan distribusi
"""

# Bandingkan distribusi statistik untuk setiap kolom
for col in data.columns:
    print(f"\nKolom: {col}")
    print("Data Asli - Mean:", data[col].mean(), "Std:", data[col].std())
    print("Data Sintetik - Mean:", data_fake[col].mean(), "Std:", data_fake[col].std())
    # Uji KS (Kolmogorov-Smirnov)
    print("KS Test p-value:", stats.ks_2samp(data[col], data_fake[col])[1])

"""Hasil Analisis dan Interpretasi

1.   Perbedaan Statistik yang Ekstrim
*   Terdapat perbedaan sangat besar antara data asli dan sintetik di semua kolom
*   Contoh
  *   Kolom `gender`: Mean asli 0.41 vs sintetik 4.18
  *   Kolom `age`: Mean asli 41.89 vs sintetik 4.22
  *   Kolom `HbA1c_level`: Mean asli 5.53 vs sintetik -2.52


2.   Hasil KS Test yang Signifikan
*   Semua kolom menunjukkan p-value = 0.0 atau sangat mendekati 0
*   Ini berarti perbedaan distribusi sangat signifikan secara statistik
*   Bahkan untuk kolom dengan perbedaan terkecil (`smoking_history`) pun p-value sangat kecil (3.02e-142)


3.   Masalah pada Data Sintetik
*   Data sintetik:
  *   Memiliki skala yang sangat berbeda
  *   Nilai mean tidak masuk akal untuk konteks variabel (misal usia 4.22 atau HbA1c negatif)
  *   Variansi (std) juga sangat berbeda

Penyebab Potensial

*   Model generatif (mungkin GAN) tidak berhasil mempelajari distribusi asli
*   Mungkin terjadi mode collapse atau masalah training lainnya
*   Data input dan output mungkin dalam skala yang berbeda
*   Tidak ada normalisasi/standarisasi yang tepat
*   Model generatif mungkin terlalu sederhana
*   Hyperparameter tidak optimal
"""

# Gabungkan data asli dan sintetik
combined = pd.concat([data.assign(Source='Real'), data_fake.assign(Source='Synthetic')])

# Lakukan PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(combined.drop('Source', axis=1))

# Visualisasikan
plt.figure(figsize=(10,6))
for source in ['Real', 'Synthetic']:
    idx = combined['Source'] == source
    plt.scatter(pca_results[idx, 0], pca_results[idx, 1], label=source, alpha=0.5)
plt.legend()
plt.title('PCA Comparison: Real vs Synthetic Data')
plt.show()

"""Interpretasi Hasil PCA

1.    Distribusi yang Sangat Berbeda
*   Data asli (Real) dan sintetik (Synthetic) menempati wilayah yang sama sekali berbeda dalam ruang PCA
*   Tidak ada overlap yang signifikan antara kedua distribusi


2.   Skala yang Tidak Kompatibel
*   Sumbu PCA menunjukkan range nilai yang ekstrim (-150 sampai 150)
*   Ini mengindikasikan perbedaan skala yang sangat besar antara fitur asli dan sintetik


3.   Struktur Data yang Berbeda
*   Data asli kemungkinan membentuk cluster yang lebih terstruktur
*   Data sintetik tersebar dengan pola yang berbeda




"""