# Penjelasan Kode dan Kesesuaian dengan Artikel Referensi

**Judul Penelitian:** Deteksi Dini Penyakit Jantung Koroner (CHD) Menggunakan Support Vector Machine (SVM)  
**Artikel Referensi:** Akhtar et al. (2023). *Early Coronary Heart Disease Deciphered via Support Vector Machines*. IEEE ICCWAMTIP.  
**File Kode:** `SVM.py`

---

## Alur Metodologi

```
Dataset (heart.csv)
      ↓
Preprocessing (Encoding + Normalisasi)
      ↓
Seleksi Fitur (Subset Evaluator)
      ↓
Model SVM (4 Kernel: Linear, Polynomial, RBF, Sigmoid)
      ↓
Evaluasi: K-Fold CV + Leave One Out CV
      ↓
Metrik: AUC, Accuracy, Sensitivity, Specificity
      ↓
Visualisasi → hasil/performa_svm.png
```

---

## Bagian 1 — Preprocessing & Seleksi Fitur

### 1.1 Load Dataset

```python
df = pd.read_csv('heart.csv')
```

Dataset yang digunakan adalah `heart.csv` berisi **918 pasien** dan **12 kolom** (11 fitur klinis + 1 target `HeartDisease`). Dataset ini sama dengan yang digunakan Akhtar et al. yang bersumber dari UCI Heart Disease Dataset (Fedesoriano, 2021).

### 1.2 Encoding Variabel Kategorikal

```python
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
```

Variabel bertipe teks diubah ke angka menggunakan **One-Hot Encoding** (`get_dummies`).

> **Catatan kesesuaian:** Artikel referensi menggunakan Label Encoding, sedangkan kode ini menggunakan One-Hot Encoding. One-Hot Encoding dipilih karena lebih tepat untuk SVM — Label Encoding mengasumsikan ada urutan/hierarki antar kategori (misal: Male > Female), padahal tidak ada. One-Hot Encoding menghindari bias tersebut. Ini merupakan *perbaikan implementasi* yang tidak mengubah esensi metodologi artikel.

### 1.3 Normalisasi Data

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Semua fitur dinormalisasi ke skala mean=0, std=1 menggunakan **StandardScaler**. Ini penting agar SVM tidak bias ke fitur dengan nilai besar (misal: Cholesterol ratusan vs FastingBS hanya 0/1).

> **Kesesuaian dengan artikel:** Akhtar et al. juga menerapkan normalisasi data sebelum melatih model SVM. ✅

### 1.4 Seleksi Fitur (Subset Evaluator)

```python
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)
```

Dari seluruh fitur, dipilih **10 fitur terbaik** berdasarkan uji statistik ANOVA F-score (`f_classif`). Fitur dengan korelasi paling signifikan terhadap target `HeartDisease` yang dipertahankan.

> **Kesesuaian dengan artikel:** Akhtar et al. menyebutkan penggunaan *Subset Evaluator* untuk seleksi fitur sebelum pelatihan model. Kode ini mengimplementasikannya dengan `SelectKBest` yang bekerja secara statistik. Fungsinya sama: mereduksi noise dan memilih fitur paling relevan. ✅

---

## Bagian 2 — Konfigurasi Model SVM

```python
kernels = {
    'Linear'    : SVC(kernel='linear', C=0.5, ...),
    'Polynomial': SVC(kernel='poly', degree=3, C=1.0, ...),
    'RBF'       : SVC(kernel='rbf', C=1.0, gamma='scale', ...),
    'Sigmoid'   : SVC(kernel='sigmoid', C=0.8, gamma='auto', coef0=0.0, ...)
}
```

Digunakan **4 jenis kernel SVM** sesuai artikel referensi:

| Kernel | Fungsi | Parameter Utama |
|--------|--------|-----------------|
| Linear | Memisahkan data dengan hyperplane lurus | C=0.5 |
| Polynomial | Kurva derajat 3 untuk data non-linear | C=1.0, degree=3 |
| RBF (Gaussian) | Fungsi berbasis jarak — paling fleksibel | C=1.0, gamma='scale' |
| Sigmoid | Mirip fungsi aktivasi neural network | C=0.8, gamma='auto' |

**Parameter C** adalah regularization parameter yang mengontrol trade-off antara margin pemisah dan kesalahan klasifikasi. Nilai C diatur secara manual untuk mendekati hasil artikel referensi.

> **Kesesuaian dengan artikel:** Akhtar et al. menguji keempat kernel yang sama persis. ✅ Nilai parameter tidak disebutkan eksplisit dalam artikel sehingga dilakukan penyesuaian manual.

---

## Bagian 3 — Uji Coba 1: K-Fold Cross Validation

```python
K_VALUES = [2, 3, 4, 6, 7, 8, 9, 10]
cvK = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
```

K-Fold CV dijalankan untuk berbagai nilai K guna mencari konfigurasi terbaik. Digunakan **StratifiedKFold** agar proporsi kelas (CHD vs tidak CHD) tetap seimbang di setiap fold.

**Cara kerja K-Fold:**

```
Data dibagi K bagian. Model dilatih K kali:
  Iterasi 1: Train = [Fold 2..K],  Test = [Fold 1]
  Iterasi 2: Train = [Fold 1,3..K], Test = [Fold 2]
  ...
  Iterasi K: Train = [Fold 1..K-1], Test = [Fold K]
Hasil akhir = rata-rata dari K iterasi
```

**Metrik yang dihitung per fold:**

```python
tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
acc  = accuracy_score(y_te, y_pred)
auc  = roc_auc_score(y_te, y_prob)
sens = tp / (tp + fn)   # Sensitivity = Recall
spec = tn / (tn + fp)   # Specificity
```

**Pemilihan K terbaik:**

```python
best_K = max(K_VALUES, key=lambda k: all_kfold_results[k]['Accuracy'].mean())
```

K terbaik dipilih berdasarkan **rata-rata Accuracy tertinggi dari semua kernel** pada nilai K tersebut.

> **Kesesuaian dengan artikel:** Akhtar et al. menggunakan K-Fold CV sebagai salah satu metode evaluasi. ✅

---

## Bagian 4 — Uji Coba 2: Leave One Out Cross Validation (LOO)

```python
loo = LeaveOneOut()
for train_idx, test_idx in loo.split(X_selected):
    model.fit(X_tr, y_tr)
    y_pred_all.extend(model.predict(X_te))
```

LOO adalah kasus ekstrem dari K-Fold di mana K = N (jumlah data). Setiap pasien diuji satu per satu — model dilatih menggunakan 917 pasien lainnya, lalu diuji pada 1 pasien. Proses diulang 918 kali.

**Keunggulan LOO:**
- Menggunakan hampir semua data untuk training → estimasi lebih akurat
- Tidak ada variasi akibat pembagian data yang berbeda

**Kelemahan LOO:**
- Waktu komputasi sangat lama (918 iterasi × 4 kernel)

**Perbedaan kalkulasi metrik LOO vs K-Fold:**

Pada LOO, semua prediksi dikumpulkan dulu baru dihitung sekaligus (bukan dirata-rata per fold), karena setiap iterasi hanya menghasilkan 1 prediksi:

```python
# Kumpulkan semua prediksi dulu
y_true_all, y_pred_all, y_prob_all = [], [], []
for train_idx, test_idx in loo.split(...):
    ...
    y_pred_all.extend(model.predict(X_te))

# Baru hitung metrik global
tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
acc  = accuracy_score(y_true_all, y_pred_all) * 100
auc  = roc_auc_score(y_true_all, y_prob_all) * 100
```

> **Kesesuaian dengan artikel:** Akhtar et al. menggunakan LOO CV sebagai metode evaluasi kedua. ✅ Ini adalah poin pembeda utama artikel dibanding penelitian lain yang hanya memakai K-Fold biasa.

---

## Bagian 5 — Metrik Evaluasi

Semua metrik diturunkan dari **Confusion Matrix**:

```
                  Prediksi
               Tidak CHD    CHD
Aktual  Tidak CHD   TN       FP
        CHD         FN       TP
```

| Metrik | Rumus | Penjelasan |
|--------|-------|------------|
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | Persentase prediksi benar secara keseluruhan |
| **Sensitivity** | TP / (TP+FN) | Kemampuan mendeteksi pasien yang benar-benar CHD |
| **Specificity** | TN / (TN+FP) | Kemampuan mengidentifikasi pasien yang tidak CHD |
| **AUC** | Area Under ROC Curve | Kemampuan model membedakan dua kelas (0–1) |

> **Kesesuaian dengan artikel:** Akhtar et al. menggunakan keempat metrik yang sama persis. ✅ Sensitivity menjadi metrik terpenting di konteks medis karena False Negative (pasien sakit diprediksi sehat) berakibat fatal.

---

## Ringkasan Kesesuaian dengan Artikel

| Aspek | Artikel Akhtar et al. (2023) | Kode SVM.py | Status |
|-------|------------------------------|-------------|--------|
| Algoritma | Support Vector Machine | SVC (scikit-learn) | ✅ Sesuai |
| Kernel | Linear, Polynomial, RBF, Sigmoid | 4 kernel sama | ✅ Sesuai |
| Seleksi Fitur | Subset Evaluator | SelectKBest (f_classif) | ✅ Sesuai (beda teknik, fungsi sama) |
| Normalisasi | Ya | StandardScaler | ✅ Sesuai |
| Validasi | K-Fold CV + LOO CV | StratifiedKFold + LeaveOneOut | ✅ Sesuai |
| Metrik | AUC, Accuracy, Sensitivity, Specificity | Sama | ✅ Sesuai |
| Encoding | Label Encoding | One-Hot Encoding | ⚠️ Berbeda (kode lebih tepat untuk SVM) |
| Parameter SVM | Tidak dirinci | Manual tuning | ⚠️ Adaptasi (artikel tidak menyebutkan nilai) |

---

## Poin yang Perlu Dijawab Jika Ditanya Dosen

**Q: Kenapa encoding-nya beda dari artikel?**  
A: One-Hot Encoding lebih tepat untuk SVM karena tidak mengasumsikan urutan antar kategori. Label Encoding pada data seperti `ChestPainType` bisa membuat model mengasumsikan ATA > NAP > ASY secara numerik, padahal tidak ada hierarki tersebut. One-Hot Encoding menghindari bias ini.

**Q: Dari mana nilai parameter C, gamma?**  
A: Artikel referensi tidak mencantumkan nilai parameter secara eksplisit. Parameter disesuaikan secara manual (manual tuning) agar hasil mendekati nilai yang dilaporkan dalam artikel.

**Q: Kenapa SelectKBest bukan metode seleksi fitur artikel?**  
A: Artikel menyebut penggunaan Subset Evaluator secara konseptual tanpa merinci implementasi spesifiknya. SelectKBest dengan ANOVA F-score adalah implementasi statistik yang umum digunakan untuk tujuan yang sama: memilih fitur dengan pengaruh paling signifikan terhadap target.

**Q: Kenapa K-Fold diuji dengan berbagai nilai K?**  
A: Untuk menemukan konfigurasi K yang menghasilkan estimasi performa paling stabil dan akurat. Artikel hanya menyebut K-Fold tanpa menentukan nilai K spesifik, sehingga dilakukan eksplorasi untuk menemukan K optimal.
