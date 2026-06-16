# Tutorial Progress 2 — Deteksi CHD dengan SVM

**Judul Penelitian:** Deteksi Dini Penyakit Jantung Koroner (CHD) Menggunakan Support Vector Machine (SVM)  
**Referensi:** Akhtar et al., 2023 IEEE ICCWAMTIP  
**Dataset:** `heart_rapi.xlsx` — 918 pasien, 12 atribut

---

## Ringkasan Metode (Baca Ini Dulu Sebelum Presentasi)

### Metode apa yang dipakai?

Penelitian ini memakai **Support Vector Machine (SVM)** — sebuah algoritma _machine learning_ untuk **klasifikasi**. Tugasnya: memprediksi apakah seorang pasien **mengidap CHD (1)** atau **tidak (0)** berdasarkan 11 data klinis (umur, jenis kelamin, tekanan darah, kolesterol, detak jantung, dll).

Diuji **4 jenis kernel SVM** — Linear, Polynomial, RBF, dan Sigmoid — lalu dibandingkan untuk mencari yang paling akurat.

Untuk menguji performa model, dipakai **3 metode validasi**:
- **Train-Test Split (80/20)** — data dibagi sekali: 80% untuk melatih, 20% untuk menguji (paling sederhana & cepat, sebagai baseline).
- **5-Fold Cross Validation** — data dibagi 5, model diuji bergiliran (lebih objektif).
- **Leave One Out (LOO)** — tiap pasien diuji satu per satu (lebih akurat, lebih lama).

### Kenapa pakai SVM?

| Alasan | Penjelasan |
|--------|-----------|
| **Cocok untuk data medis** | SVM terbukti di banyak literatur kedokteran untuk klasifikasi diagnosis penyakit. |
| **Kuat di data berdimensi tinggi** | Dataset ini punya 11 fitur — SVM tetap stabil meski fiturnya banyak. |
| **Tahan overfitting** | SVM mencari _margin_ pemisah terlebar antar kelas, sehingga tidak gampang "hafalan" data latih. |
| **Mengikuti artikel referensi** | Akhtar et al. (2023) memakai SVM untuk kasus yang sama, jadi metode ini bisa kita bandingkan langsung. |

### Kenapa pakai 3 metode (Train-Test Split → CV → LOO)?

Kita mulai dari yang paling sederhana lalu naik ke yang paling ketat:

- **Train-Test Split (80/20)** adalah baseline standar — bagi data sekali, latih di 80%, uji di 20%. Cepat dan mudah dipahami, tapi hasilnya bisa **kebetulan bagus/jelek** tergantung data mana yang kebetulan masuk ke bagian uji.
- **Cross Validation** menutup kelemahan itu: model diuji berkali-kali pada bagian data yang berbeda lalu dirata-rata — sehingga **lebih objektif dan tidak bias**.
- **LOO** adalah bentuk CV paling ketat (tiap pasien jadi data uji satu per satu).

Membandingkan ketiganya membuktikan bahwa hasil model benar-benar stabil, bukan kebetulan. Bukti nyatanya ada di hasil: pada Train-Test Split tunggal, kernel **RBF** terlihat paling unggul, tetapi setelah diuji lewat CV dan LOO, justru **Linear & Sigmoid** yang konsisten terbaik — inilah alasan kenapa satu split saja tidak cukup.

---

## Daftar Isi

1. [Persiapan](#1-persiapan)
2. [Cara Menjalankan](#2-cara-menjalankan)
3. [Output yang Dihasilkan](#3-output-yang-dihasilkan)
4. [Penjelasan Kode](#4-penjelasan-kode)
5. [Penjelasan Hasil](#5-penjelasan-hasil)
6. [Cara Presentasi](#6-cara-presentasi)

---

## 1. Persiapan

Pastikan `uv` sudah terinstall. Jika belum:
> - **Windows:** `pip install uv`
> - **Mac/Linux:** `pip3 install uv` atau `brew install uv` (jika pakai Homebrew)

Lalu masuk ke folder project terlebih dahulu:

**Windows:**
```bash
cd "C:\Users\Hariz\Downloads\ML TEO"
```

**Mac/Linux:**
```bash
cd "/Users/hariz/Desktop/CODE PROJECT/ML TEO"
```

**Langkah 1 — Buat virtual environment**

`uv pip install` WAJIB punya virtual environment dulu. Buat dengan:

```bash
uv venv
```

> Tanpa langkah ini, `uv pip install` akan gagal dengan error:
> `No virtual environment found; run 'uv venv' to create an environment`

**Kenapa harus pakai virtual environment?**

Virtual environment (`.venv`) adalah folder Python terpisah khusus untuk satu project. Alasannya:

- **Isolasi antar project** — Library yang diinstall (pandas, scikit-learn, dll) cuma masuk ke folder `.venv` project ini, tidak mengotori Python sistem atau project lain. Project A bisa pakai scikit-learn versi 1.9, project B versi 1.0, tanpa bentrok.
- **Mencegah konflik versi** — Tanpa venv, semua library numpuk jadi satu. Kalau dua project butuh versi berbeda dari library yang sama, salah satu pasti rusak.
- **Tidak butuh akses admin** — Install ke `.venv` tidak perlu `sudo`/administrator, karena cuma menulis ke folder project sendiri.
- **Mudah direset** — Kalau ada yang rusak, cukup hapus folder `.venv` lalu `uv venv` lagi — Python sistem tetap aman.
- **`uv` mewajibkannya** — Makanya `uv pip install` langsung menolak kalau belum ada `.venv` (beda dengan `pip` biasa yang diam-diam install ke sistem dan sering bikin berantakan).

**Langkah 2 — Install library**

```bash
uv pip install pandas scikit-learn matplotlib seaborn openpyxl
```

> Catatan: Command di atas sama di semua OS. Jika `.venv` terhapus, ulangi `uv venv` lalu install lagi.

Pastikan struktur folder seperti ini:

```
ML TEO/
├── heart_rapi.xlsx          ← dataset
├── progress2_svm_chd.py     ← kode utama
└── tutorial.md              ← file ini
```

---

## 2. Cara Menjalankan

**Langkah 1 — Buka terminal**

Di VS Code: tekan `Ctrl + backtick` untuk membuka terminal

**Langkah 2 — Masuk ke folder**

```bash
cd "C:\Users\Hariz\Downloads\ML TEO"
```

**Langkah 3 — Jalankan kode**

```bash
python progress2_svm_chd.py
```

**Langkah 4 — Tunggu proses selesai**

Proses berjalan dalam 3 tahap:
- **Train-Test Split (80/20)** → instan (sekali bagi data)
- **5-Fold CV** → selesai dalam ~10 detik
- **Leave One Out CV** → butuh ~3–5 menit (918 iterasi, normal)

Setelah selesai akan muncul 3 jendela grafik secara otomatis.

---

## 3. Output yang Dihasilkan

| File | Isi |
|------|-----|
| `hasil_performa_svm.png` | Grafik perbandingan Accuracy, AUC, Sensitivity, Specificity untuk 4 kernel SVM (3 panel: Train-Test Split, 5-Fold, dan LOO) |
| `heatmap_korelasi.png` | Heatmap korelasi antar fitur dataset |
| `confusion_matrix_sigmoid.png` | Confusion matrix model terbaik (Sigmoid) |

Di terminal juga akan tercetak **Tabel 1**, **Tabel 2**, dan **Tabel 3** berisi nilai numerik lengkap.

---

## 4. Penjelasan Kode

### Bagian 1 — Preprocessing Data

```python
df = pd.read_excel('heart_rapi.xlsx')
```
Membaca dataset. Dataset berisi 918 baris (pasien) dan 12 kolom (fitur + target).

```python
le = LabelEncoder()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col])
```
Variabel kategorikal (teks) diubah ke angka karena SVM hanya menerima input numerik.
Contoh: `Sex` → Male = 1, Female = 0

```python
X_scaled = scaler.fit_transform(X)
```
Normalisasi agar semua fitur punya skala yang sama (mean=0, std=1). Ini penting supaya SVM tidak bias ke fitur dengan nilai besar (seperti Cholesterol yang ratusan vs FastingBS yang hanya 0/1).

---

### Bagian 2 — Model SVM

```python
kernels = {
    'Linear'    : SVC(kernel='linear',  ...),
    'Polynomial': SVC(kernel='poly',    ...),
    'RBF'       : SVC(kernel='rbf',     ...),
    'Sigmoid'   : SVC(kernel='sigmoid', ...),
}
```

Digunakan 4 jenis kernel SVM sesuai artikel referensi:

| Kernel | Cara Kerja | Keunggulan |
|--------|-----------|-----------|
| **Linear** | Memisahkan data dengan garis lurus | Sederhana, cepat, interpretable |
| **Polynomial** | Memisahkan dengan kurva derajat 3 | Baik untuk data non-linear |
| **RBF** | Menggunakan fungsi Gaussian | Paling fleksibel, populer |
| **Sigmoid** | Mirip fungsi aktivasi neural network | Baik untuk data biner |

---

### Bagian 3 — Uji Coba: Train-Test Split (80/20)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
```

**Cara kerja Train-Test Split:**
- Data dibagi **sekali** menjadi dua bagian: 80% untuk **melatih** model dan 20% untuk **menguji**.
- `stratify=y` menjaga proporsi kelas (CHD vs Tidak CHD) tetap sama di kedua bagian, supaya pembagian adil.
- `random_state=42` membuat pembagian selalu sama setiap dijalankan (bisa direproduksi).

```
Total: 918 pasien
┌──────────────────────────────┬──────────┐
│      Data Latih (734)        │ Uji(184) │
│            80%               │   20%    │
└──────────────────────────────┴──────────┘
```

Model dilatih pada 734 pasien, lalu diuji pada 184 pasien yang **belum pernah dilihat** model. Ini metode paling sederhana dan jadi titik awal (baseline) sebelum metode validasi yang lebih ketat.

---

### Bagian 4 — Uji Coba: 5-Fold Cross Validation

```python
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = cross_val_score(model, X_scaled, y, cv=cv5, scoring='accuracy')
```

**Cara kerja 5-Fold CV:**
- Dataset dibagi 5 bagian sama besar
- Model dilatih 5 kali, setiap kali menggunakan 4 bagian untuk training dan 1 bagian untuk testing
- Hasil akhir = rata-rata dari 5 iterasi

```
Data: [A][B][C][D][E]

Iterasi 1: Train=[B,C,D,E]  Test=[A]
Iterasi 2: Train=[A,C,D,E]  Test=[B]
Iterasi 3: Train=[A,B,D,E]  Test=[C]
Iterasi 4: Train=[A,B,C,E]  Test=[D]
Iterasi 5: Train=[A,B,C,D]  Test=[E]
```

---

### Bagian 5 — Uji Coba: Leave One Out (LOO)

```python
loo = LeaveOneOut()
for train_idx, test_idx in loo.split(X_scaled):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
```

**Cara kerja LOO:**
- Setiap pasien digunakan sebagai data test satu per satu
- Model dilatih menggunakan 917 pasien lainnya, lalu diuji pada 1 pasien
- Proses diulang 918 kali (sebanyak jumlah data)
- Lebih akurat dari 5-Fold tapi lebih lambat

---

### Bagian 6 — Metrik Evaluasi

Dihitung dari **Confusion Matrix**:

```
               Prediksi
               Tidak CHD   CHD
Aktual Tidak CHD   TN        FP
       CHD         FN        TP
```

| Metrik | Rumus | Arti |
|--------|-------|------|
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | Seberapa sering model benar secara keseluruhan |
| **Sensitivity** | TP / (TP+FN) | Seberapa baik model mendeteksi pasien yang benar-benar CHD |
| **Specificity** | TN / (TN+FP) | Seberapa baik model mengidentifikasi pasien yang tidak CHD |
| **AUC** | Area Under ROC Curve | Kemampuan model membedakan kedua kelas (0–1, makin tinggi makin baik) |

> **Catatan:** Sensitivity sangat penting di konteks medis — kita tidak ingin model "melewatkan" pasien yang sakit (False Negative berbahaya).

---

## 5. Penjelasan Hasil

### Tabel 1 — Hasil Train-Test Split (80/20)

| Model | AUC | Accuracy | Sensitivity | Specificity |
|-------|-----|----------|-------------|-------------|
| Linear | ~89.1% | ~87.0% | ~93.1% | ~79.3% |
| Polynomial | ~89.7% | ~88.0% | ~91.2% | ~84.2% |
| **RBF** | **~92.9%** | **~89.1%** | **~95.1%** | ~81.7% |
| Sigmoid | ~86.4% | ~80.4% | ~88.2% | ~70.7% |

> Pada split tunggal ini, **RBF** terlihat paling unggul (Accuracy ~89.1%, AUC ~92.9%). Namun hasil ini bergantung pada data uji yang kebetulan terpilih — itulah kenapa kita lanjut ke Cross Validation untuk validasi yang lebih objektif.

### Tabel 2 — Hasil 5-Fold Cross Validation

| Model | AUC | Accuracy | Sensitivity | Specificity |
|-------|-----|----------|-------------|-------------|
| Linear | ~91.8% | ~85.6% | ~85.6% | ~83.8% |
| Polynomial | ~90.0% | ~84.2% | ~84.2% | ~83.2% |
| RBF | ~91.0% | ~83.7% | ~83.7% | ~83.2% |
| **Sigmoid** | **~90.6%** | **~86.4%** | **~86.4%** | **~85.3%** |

### Tabel 3 — Hasil Leave One Out CV

| Model | AUC | Accuracy | Sensitivity | Specificity |
|-------|-----|----------|-------------|-------------|
| **Linear** | **~93.6%** | ~87.7% | ~87.7% | ~87.2% |
| Polynomial | ~89.9% | ~81.6% | ~81.6% | ~87.2% |
| RBF | ~89.4% | ~83.3% | ~83.3% | ~83.5% |
| **Sigmoid** | ~93.5% | **~87.8%** | **~87.8%** | **~87.4%** |

**Kesimpulan:**
- Pada **Train-Test Split**, RBF terlihat unggul — tapi ini hasil dari satu pembagian saja, belum tentu mewakili.
- Setelah divalidasi lebih ketat (5-Fold & LOO), **Sigmoid** punya Accuracy tertinggi (~87.8% di LOO) → dipilih sebagai model terbaik.
- Model **Linear** punya AUC tertinggi (~93.6%) → paling baik dalam membedakan kelas.
- Metode LOO secara konsisten menghasilkan nilai lebih tinggi dibanding 5-Fold.
- **Pelajaran penting:** model "terbaik" pada satu split (RBF) berbeda dari model terbaik setelah validasi menyeluruh (Sigmoid/Linear) — membuktikan kenapa CV diperlukan.

---

## 6. Cara Presentasi

### Pembukaan
> "Pada Progress 2 ini, kami mengimplementasikan metodologi penelitian dan melakukan uji coba model SVM untuk deteksi dini Coronary Heart Disease (CHD) mengikuti artikel referensi Akhtar et al. 2023."

### Jelaskan Metodologi (urut)
1. Dataset → 918 pasien, 12 fitur klinis
2. Preprocessing → encoding + normalisasi
3. Train-Test Split → bagi data 80% latih / 20% uji
4. Model → SVM 4 kernel (Linear, Poly, RBF, Sigmoid)
5. Evaluasi → Train-Test Split + 5-Fold CV + Leave One Out CV

### Tunjukkan Hasil
- Tampilkan `hasil_performa_svm.png` → bandingkan 4 kernel
- Tampilkan `heatmap_korelasi.png` → jelaskan hubungan antar fitur
- Sebutkan model terbaik: Sigmoid dengan Accuracy 87.8% dan AUC 93.5%

### Penutup
> "Hasil yang diperoleh konsisten dengan artikel referensi. Model SVM Sigmoid terbukti paling efektif untuk klasifikasi CHD pada dataset ini."

---

### Pertanyaan yang Mungkin Ditanya

**Q: Kenapa pakai SVM?**  
A: SVM efektif untuk data berdimensi tinggi, tahan terhadap overfitting, dan sudah terbukti di banyak literatur medis untuk tugas klasifikasi.

**Q: Kenapa ada 4 kernel?**  
A: Setiap kernel punya cara berbeda dalam memisahkan data. Kita bandingkan semua untuk menemukan yang paling cocok dengan karakteristik dataset ini.

**Q: Kenapa LOO lebih baik dari 5-Fold?**  
A: LOO menggunakan hampir semua data (n-1 sampel) untuk training di setiap iterasi, sehingga estimasi performanya lebih akurat. Kelemahannya hanya di waktu komputasi yang lebih lama.

**Q: Kenapa pakai Train-Test Split kalau sudah ada Cross Validation?**  
A: Train-Test Split adalah baseline standar yang paling mudah dipahami dan cepat. Kita pakai sebagai titik awal, lalu naik ke CV dan LOO yang lebih ketat. Membandingkan ketiganya justru menunjukkan kelebihan CV: pada split tunggal RBF terlihat terbaik, tapi setelah validasi menyeluruh ternyata Sigmoid/Linear yang konsisten unggul.

**Q: Apa artinya Sensitivity tinggi?**  
A: Model jarang "melewatkan" pasien yang benar-benar sakit CHD. Ini sangat penting di dunia medis karena False Negative (pasien sakit tapi diprediksi sehat) bisa berakibat fatal.
