# 🔍 SentimenAja: Analisis Sentimen Opini Publik terhadap Polri di Platform X (Twitter)

> **Produk Luaran Tugas Akhir / Skripsi**  
> Program Studi S1 Teknik Informatika — Fakultas Ilmu Komputer  
> **Universitas Dian Nuswantoro (UDINUS) Semarang** (2026)  
> **Peneliti:** Ariella Risqy Maulana (NIM: A11.2022.14035)  
> **Dosen Pembimbing:** Yani Parti Astuti, M.Kom  

---

## 📑 Daftar Isi
- [Ringkasan Eksekutif & Abstrak Skripsi](#-ringkasan-eksekutif--abstrak-skripsi)
- [Latar Belakang & Identifikasi Masalah](#-latar-belakang--identifikasi-masalah)
- [Metodologi Penelitian & Pipeline Sistem](#-metodologi-penelitian--pipeline-sistem)
- [Temuan & Hasil Komparasi Model](#-temuan--hasil-komparasi-model)
- [Tech Stack & Arsitektur Teknologi](#-tech-stack--arsitektur-teknologi)
- [Modul & Fitur Aplikasi](#-modul--fitur-aplikasi)
- [Panduan Instalasi & Menjalankan Aplikasi](#-panduan-instalasi--menjalankan-aplikasi)
- [Struktur Direktori](#-struktur-direktori)
- [Sitasi & Hak Cipta](#-sitasi--hak-cipta)

---

## 🎓 Ringkasan Eksekutif & Abstrak Skripsi

### Judul Skripsi:
> **"ANALISIS SENTIMEN TERHADAP KEPOLISIAN NEGARA REPUBLIK INDONESIA PERIODE 2022 - 2025 MENGGUNAKAN PERBANDINGAN MULTINOMIAL NAIVE BAYES DAN SUPPORT VECTOR MACHINE LINEAR"**

### Abstrak:
Kemajuan teknologi media sosial membuka ruang luas bagi publik untuk mengevaluasi kinerja pelayan masyarakat, termasuk Kepolisian Negara Republik Indonesia (Polri). Di sisi lain, tingginya intensitas publikasi negatif terkait isu kekerasan serta penyalahgunaan wewenang berisiko mereduksi kredibilitas institusi.

Penelitian ini diorientasikan untuk mengetahui sentimen masyarakat terhadap Polri di platform X periode 2022–2025 dengan mengomparasikan keandalan algoritma **Multinomial Naïve Bayes (MNB)** dan **Linear Support Vector Machine (Linear SVM)**. Pendekatan web scraping berhasil menghimpun **18.484 data mentah** yang menyisakan **14.382 data bersih** pascaproses pembersihan. Anotasi sentimen berbasis model deep learning **IndoBERTweet** mengidentifikasi adanya **7.969 data positif (55,4%)** dan **6.413 data negatif (44,6%)**. 

Proses ekstraksi fitur menerapkan pembobotan **TF-IDF (unigram dan bigram)** dengan pengujian pada tiga variasi proporsi data (**90:10, 80:20, dan 70:30**). Eksperimen menunjukkan bahwa **SVM Linier secara konsisten mengungguli MNB**, dengan capaian puncak pada rasio **90:10** yang mencatatkan **Akurasi 92,49%**, **Recall 92,84%**, dan **F1-score 93,18%**. Kendati demikian, MNB menonjol secara spesifik pada stabilitas nilai **Presisi di kisaran 95,65%–96,18%**. Kesimpulannya, SVM Linier terbukti lebih efektif dan minim bias dalam mengklasifikasikan tren opini publik terhadap kepolisian.

**Kata Kunci:** *Analisis Sentimen, Kepolisian Negara Republik Indonesia, IndoBERTweet, Multinomial Naïve Bayes, Support Vector Machine Linear, TF-IDF, Streamlit.*

---

## 🎯 Latar Belakang & Identifikasi Masalah

1. **Volume Data Tak Terstruktur yang Masif**: Opini publik mengenai kinerja Polri di platform X mengalir dalam volume besar, dinamis, dan tidak terstruktur. Evaluasi manual rentan terhadap bias subjektivitas serta keterbatasan waktu.
2. **Kebutuhan Pseudo-Labeling yang Akurat**: Pelabelan ribuan data tweet secara manual membutuhkan sumber daya besar. Penggunaan model Transformer bahasa informal seperti *IndoBERTweet* menjadi solusi anotasi otomatis (*pseudo-labeling*) berkualitas tinggi.
3. **Urgensi Perbandingan Algoritma Klasifikasi**: Diperlukan studi komparatif antara algoritma probabilistik (*Multinomial Naive Bayes*) dan algoritma pemisah bidang spasial (*Linear Support Vector Machine*) dalam menangani data teks berdimensi tinggi dengan variasi rasio data.
4. **Implementasi Sistem Praktis**: Menghadirkan sistem perangkat lunak berbasis antarmuka grafis (*Graphical User Interface*) yang memungkinkan pengujian dataset maupun analisis kalimat secara instan (*real-time*).

---

## 🔬 Metodologi Penelitian & Pipeline Sistem

Alur kerja metodologis yang diterapkan dalam penelitian dan diimplementasikan ke dalam aplikasi:

```mermaid
flowchart TD
    A[Data Crawling via Tweet-Harvest <br>18.484 tweet mentah] --> B[Preprocessing Pipeline]
    B --> B1[1. Text Cleaning: Hapus URL, Tagar, Mention, Simbol, Angka]
    B1 --> B2[2. Case Folding: Lowercase]
    B2 --> B3[3. Filtering: Rentang Tahun 2022-2025 & Keyword Polri]
    B3 --> B4[4. Normalization: Kamus Colloquial Indonesian Lexicon]
    B4 --> C[Data Bersih: 14.382 Baris]
    C --> D[Pelabelan Otomatis IndoBERTweet<br>Positif: 7.969 | Negatif: 6.413]
    D --> E[Stratified Data Split<br>Rasio 90:10 | 80:20 | 70:30]
    E --> F[Ekstraksi Fitur TF-IDF<br>Unigram + Bigram, Sublinear TF, Max Features 5000]
    F --> G1[Model Multinomial Naïve Bayes]
    F --> G2[Model Linear SVM]
    G1 & G2 --> H[Evaluasi Model: Confusion Matrix, Akurasi, Presisi, Recall, F1-Score]
    D --> I[Visualisasi Word Cloud & Document Frequency]
```

---

## 📊 Temuan & Hasil Komparasi Model

Evaluasi model dilakukan menggunakan *Confusion Matrix* pada 3 variasi skenario pembagian data:

| Rasio Data (Train : Test) | Algoritma Model | Akurasi (%) | Presisi (%) | Recall (%) | F1-Score (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **90 : 10** | **Linear SVM (Terbaik)** | **92,49%** | **93,55%** | **92,84%** | **93,18%** |
| *(12.943 : 1.439)* | Multinomial Naïve Bayes | 90,06% | **96,18%** | 85,44% | 90,48% |
| **80 : 20** | **Linear SVM** | **91,44%** | **92,92%** | **91,53%** | **92,20%** |
| *(11.505 : 2.877)* | Multinomial Naïve Bayes | 89,67% | **95,89%** | 85,00% | 90,10% |
| **70 : 30** | **Linear SVM** | **90,56%** | **92,39%** | **90,42%** | **91,38%** |
| *(10.067 : 4.315)* | Multinomial Naïve Bayes | 89,38% | **95,65%** | 84,69% | 89,82% |

### Analisis Hasil:
1. **Keunggulan Linear SVM**: SVM Linear mencatatkan performa terbaik di seluruh rasio data dengan F1-Score konsisten $>91\%$. Kernel linear sangat optimal untuk teks TF-IDF berdimensi tinggi yang bersifat *linearly separable*.
2. **Karakteristik Multinomial Naive Bayes**: MNB mencatatkan nilai Presisi tertinggi ($95,65\% - 96,18\%$), menandakan MNB sangat selektif dan minim menghasilkan kesalahan *False Positive*.
3. **Karakteristik Opini Publik**: 
   - **Sentimen Positif (55,4%)**: Dominan pada topik pelayanan masyarakat, patroli kamtibmas, dan implementasi program Polri Presisi.
   - **Sentimen Negatif (44,6%)**: Terpusat pada kritik tajam terkait kasus hukum viral (misal kasus Sambo), penindakan tilang, dan perilaku oknum aparat.

---

## 🛠️ Tech Stack & Arsitektur Teknologi

| Komponen | Pustaka / Tools | Peran dalam Sistem |
| :--- | :--- | :--- |
| **Bahasa Utama** | Python 3.13 / 3.9+ | Inti komputasi & logika program |
| **Web Dashboard** | Streamlit | Framework antarmuka grafis interaktif berbasis web |
| **Data Crawling** | Tweet Harvest (`npx`) | Web scraping data cuitan platform X tanpa batasan kuota API resmi |
| **Anotasi Data** | Hugging Face Transformers & PyTorch | Inferensi model deep learning `Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis` |
| **Klasifikasi & TF-IDF** | Scikit-Learn | `TfidfVectorizer`, `MultinomialNB`, `LinearSVC`, `train_test_split`, `metrics` |
| **Normalisasi Slang** | Kamus Alay (Colloquial Indonesian Lexicon) | Konversi kata non-baku (4.300+ lema) via HTTP dynamic downloader |
| **Data Wrangling** | Pandas & NumPy | Manajemen dataset tabular, I/O CSV & Excel, matriks DF |
| **Visualisasi** | Matplotlib, Seaborn, WordCloud | Visualisasi grafik batang, heatmap matriks konfusi, dan awan kata |

---

## 💻 Modul & Fitur Aplikasi

Aplikasi **SentimenAja** dibangun dengan dua mode navigasi utama:

### 1. 📊 Modul Analisis Dataset
- **Unggah Dataset & Pemilih Kolom**: Input file CSV/Excel dengan pratinjau interaktif.
- **Pipeline 10 Langkah Terpadu**:
  1. *Text Cleaning* (pembersihan noise & drop duplicate).
  2. *Case Folding* (penyeragaman huruf kecil).
  3. *Filter Keyword & Tahun* (70+ kata kunci Polri bawaan/kustom).
  4. *Normalisasi Kamus Alay* (multi-source download + opsi manual upload).
  5. *Pelabelan IndoBERTweet* (auto-labeling dengan progress bar & status GPU/CPU).
  6. *Data Splitting* (konfigurasi persentase train-test dengan stratifikasi).
  7. *TF-IDF Vectorization* (analisis Top-30 terms & document frequency kata kunci Polri).
  8. *Pelatihan Model ML* (training Naive Bayes & Linear SVM dengan tabel classification report).
  9. *Confusion Matrix Heatmap* (evaluasi visual prediksi vs aktual).
  10. *Word Cloud Sentimen* (visualisasi kata kunci positif vs negatif).

### 2. ✍️ Modul Analisis Teks Tunggal (*Real-Time Inference*)
- Input teks opini bebas dari pengguna.
- Eksekusi praproses instan (*Cleaning* $\rightarrow$ *Case Folding* $\rightarrow$ *Normalisasi*).
- Prediksi instan IndoBERTweet dilengkapi visualisasi *Confidence Score Bar*.
- Prediksi komparatif model Machine Learning (Multinomial NB & Linear SVM) beserta probabilitas kelas dan ringkasan metrik dari data uji.

---

## 🚀 Panduan Instalasi & Menjalankan Aplikasi

### 1. Persiapan Lingkungan
```bash
# Clone repositori
git clone https://github.com/AriellaRisqyM/SentimenAja.git
cd SentimenAja

# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Instalasi Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Menjalankan Dashboard
```bash
streamlit run streamlit_app.py
```
Akses dasbor aplikasi di peramban web Anda melalui: `http://localhost:8501`.

---

## 📁 Struktur Direktori

```text
SentimenAja/
├── .devcontainer/         # Konfigurasi container development VS Code
├── .github/               # Workflows & automasi repositori
├── streamlit_app.py       # Kode sumber utama aplikasi Streamlit
├── requirements.txt       # Daftar dependensi & paket pustaka Python
├── README.md              # Dokumentasi lengkap sistem dan skripsi
├── LICENSE                # Lisensi Apache-2.0
└── .gitignore             # File pengecualian Git
```

---

## 📄 Sitasi & Hak Cipta

Karya ini merupakan luaran penelitian Tugas Akhir / Skripsi di Universitas Dian Nuswantoro Semarang.

```bibtex
@article{maulana2026analisissentimen,
  author    = {Ariella Risqy Maulana},
  title     = {Analisis Sentimen Terhadap Kepolisian Negara Republik Indonesia Periode 2022 - 2025 Menggunakan Perbandingan Multinomial Naive Bayes Dan Support Vector Machine Linear},
  school    = {Fakultas Ilmu Komputer, Universitas Dian Nuswantoro Semarang},
  year      = {2026}
}
```

Didistribusikan di bawah lisensi [Apache License 2.0](LICENSE).
