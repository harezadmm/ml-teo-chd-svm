# Tutorial Progress 2 — Deteksi CHD dengan SVM

**Judul Penelitian:** Deteksi Dini Penyakit Jantung Koroner (CHD) Menggunakan Support Vector Machine (SVM)  
**Referensi:** Akhtar et al., 2023 IEEE ICCWAMTIP  
**Dataset:** `heart_rapi.xlsx` — 918 pasien, 12 atribut

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

Pastikan semua library sudah terinstall. Jalankan perintah berikut di terminal:

**Windows:**
```bash
uv pip install pandas scikit-learn matplotlib seaborn openpyxl
```

**Mac/Linux:**
```bash
uv pip install pandas scikit-learn matplotlib seaborn openpyxl
```

> Catatan: Command `uv pip install` sama di semua OS. Pastikan `uv` sudah terinstall — jika belum, install dulu dengan:
> - **Windows:** `pip install uv`
> - **Mac/Linux:** `pip3 install uv` atau `brew install uv` (jika pakai Homebrew)

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

Proses berjalan dalam 2 tahap:
- **5-Fold CV** → selesai dalam ~10 detik
- **Leave One Out CV** → butuh ~3–5 menit (918 iterasi, normal)

Setelah selesai akan muncul 3 jendela grafik secara otomatis.

---

## 3. Output yang Dihasilkan

| File | Isi |
|------|-----|
| `hasil_performa_svm.png` | Grafik perbandingan Accuracy, AUC, Sensitivity, Specificity untuk 4 kernel SVM (5-Fold dan LOO) |
| `heatmap_korelasi.png` | Heatmap korelasi antar fitur dataset |
| `confusion_matrix_sigmoid.png` | Confusion matrix model terbaik (Sigmoid) |

Di terminal juga akan tercetak **Tabel 2** dan **Tabel 3** berisi nilai numerik lengkap.

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

### Bagian 3 — Uji Coba: 5-Fold Cross Validation

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

### Bagian 4 — Uji Coba: Leave One Out (LOO)

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

### Bagian 5 — Metrik Evaluasi

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
- Model **Sigmoid** punya Accuracy tertinggi (~87.8%) di LOO → dipilih sebagai model terbaik
- Model **Linear** punya AUC tertinggi (~93.6%) → paling baik dalam membedakan kelas
- Metode LOO secara konsisten menghasilkan nilai lebih tinggi dibanding 5-Fold

---

## 6. Cara Presentasi

### Pembukaan
> "Pada Progress 2 ini, kami mengimplementasikan metodologi penelitian dan melakukan uji coba model SVM untuk deteksi dini Coronary Heart Disease (CHD) mengikuti artikel referensi Akhtar et al. 2023."

### Jelaskan Metodologi (urut)
1. Dataset → 918 pasien, 12 fitur klinis
2. Preprocessing → encoding + normalisasi
3. Model → SVM 4 kernel (Linear, Poly, RBF, Sigmoid)
4. Evaluasi → 5-Fold CV + Leave One Out CV

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

**Q: Apa artinya Sensitivity tinggi?**  
A: Model jarang "melewatkan" pasien yang benar-benar sakit CHD. Ini sangat penting di dunia medis karena False Negative (pasien sakit tapi diprediksi sehat) bisa berakibat fatal.
