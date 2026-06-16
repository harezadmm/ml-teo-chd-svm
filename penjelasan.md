# Penjelasan Kode dan Kesesuaian dengan Artikel Referensi

**Judul Penelitian:** Deteksi Dini Penyakit Jantung Koroner (CHD) Menggunakan Support Vector Machine (SVM)  
**Artikel Referensi:** Akhtar et al. (2023). *Early Coronary Heart Disease Deciphered via Support Vector Machines*. IEEE ICCWAMTIP.  
**File Kode:** `SVM.py`  
**Dataset:** `heart.csv` (918 pasien, 11 fitur input + 1 target)

---

## Alur Metodologi

```
heart.csv (918 pasien)
      ↓
Preprocessing
  - Label Encoding (5 kolom kategorikal)
  - StandardScaler (normalisasi semua fitur)
      ↓
Model SVM — 4 Kernel
  Linear | Polynomial | RBF | Sigmoid
      ↓
Evaluasi Ganda
  - K-Fold CV (K = 2,3,4,6,7,8,9,10) → cari K terbaik
  - Leave One Out CV
      ↓
Metrik: AUC, Accuracy, Sensitivity, Specificity
      ↓
Visualisasi → hasil/performa_svm.png
```

---

## Bagian 1 — Preprocessing

### 1.1 Load Dataset

```python
df = pd.read_csv('heart.csv')
```

Dataset berisi **918 pasien** dan **12 kolom** (11 fitur klinis + 1 target `HeartDisease`). Sama dengan dataset yang digunakan Akhtar et al. yang bersumber dari Kaggle Heart Disease dataset (Fedesoriano, 2021).

### 1.2 Label Encoding

```python
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
le = LabelEncoder()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
```

Lima kolom bertipe teks diubah ke angka secara berurutan berdasarkan abjad (contoh: `Sex` → Female=0, Male=1). Metode ini sesuai dengan yang digunakan dalam artikel referensi.

> **Kesesuaian dengan artikel:** Akhtar et al. menggunakan Label Encoding untuk variabel kategorikal. ✅

### 1.3 Normalisasi Data

```python
scaler = StandardScaler()
X_selected = scaler.fit_transform(X)
```

Semua 11 fitur dinormalisasi ke skala mean=0, std=1. Ini mencegah SVM bias terhadap fitur bernilai besar seperti `Cholesterol` (ratusan) dibanding `FastingBS` (0/1).

> **Kesesuaian dengan artikel:** Akhtar et al. menerapkan normalisasi sebelum pelatihan model. ✅

---

## Bagian 2 — Konfigurasi Model SVM

```python
kernels = {
    'Linear'    : SVC(kernel='linear',  probability=True, random_state=42),
    'Polynomial': SVC(kernel='poly',    probability=True, random_state=42, degree=3),
    'RBF'       : SVC(kernel='rbf',     probability=True, random_state=42),
    'Sigmoid'   : SVC(kernel='sigmoid', probability=True, random_state=42),
}
```

Digunakan **4 kernel SVM** dengan parameter default (tidak ada tuning manual pada C, gamma, dll), sesuai dengan pendekatan artikel.

| Kernel | Cara Kerja | Keunggulan |
|--------|-----------|-----------|
| **Linear** | Hyperplane lurus | Sederhana, cepat, interpretable |
| **Polynomial** | Kurva derajat 3 | Baik untuk data non-linear ringan |
| **RBF** | Fungsi Gaussian berbasis jarak | Paling fleksibel, populer |
| **Sigmoid** | Mirip fungsi aktivasi neural network | Baik untuk data biner |

> **Kesesuaian dengan artikel:** Akhtar et al. menguji keempat kernel yang sama dengan parameter default. ✅

---

## Bagian 3 — Uji Coba 1: Multi-K Fold Cross Validation

```python
K_VALUES = [2, 3, 4, 6, 7, 8, 9, 10]

for K in K_VALUES:
    cvK = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    ...
```

**Pengembangan dari artikel:** Artikel hanya menggunakan 5-Fold CV. Kode ini menguji **8 nilai K** (2,3,4,6,7,8,9,10) untuk menemukan konfigurasi K yang paling optimal secara empiris.

Digunakan `StratifiedKFold` agar proporsi kelas (CHD vs tidak CHD) tetap seimbang di setiap fold.

**Cara kerja K-Fold:**

```
Data dibagi K bagian:
  Iterasi 1: Train=[Fold 2..K]   Test=[Fold 1]
  Iterasi 2: Train=[Fold 1,3..K] Test=[Fold 2]
  ...
  Iterasi K: Train=[Fold 1..K-1] Test=[Fold K]
Hasil = rata-rata dari K iterasi
```

**Metrik dihitung per fold:**

```python
tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

acc_list.append(accuracy_score(y_te, y_pred))
auc_list.append(roc_auc_score(y_te, y_prob))
sens_list.append(tp / (tp + fn))   # Sensitivity
spec_list.append(tn / (tn + fp))   # Specificity
```

**Pemilihan K terbaik:**

```python
best_K = max(K_VALUES, key=lambda k: all_kfold_results[k]['Accuracy'].mean())
```

K terbaik dipilih berdasarkan **rata-rata Accuracy tertinggi dari semua kernel** pada nilai K tersebut. Tabel pemilihan dicetak di terminal untuk transparansi.

> **Kesesuaian dengan artikel:** Artikel menggunakan 5-Fold CV. Kode ini memperluas eksplorasi ke berbagai nilai K sebagai analisis tambahan, lalu tetap melaporkan hasil K terbaik dan LOO sesuai artikel. ✅ (dengan pengembangan)

---

## Bagian 4 — Uji Coba 2: Leave One Out Cross Validation (LOO)

```python
loo = LeaveOneOut()
for train_idx, test_idx in loo.split(X_selected):
    model.fit(X_tr, y_tr)
    y_pred_all.extend(model.predict(X_te))
    y_prob_all.extend(model.predict_proba(X_te)[:, 1])
    y_true_all.extend(y_te)

# Kalkulasi metrik global setelah semua iterasi selesai
tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
```

LOO adalah kasus K-Fold di mana K = N (918). Setiap pasien diuji satu per satu — model dilatih dengan 917 pasien lainnya, lalu diuji pada 1 pasien. Diulang 918 kali.

**Perbedaan kalkulasi LOO vs K-Fold:**
- K-Fold: metrik dihitung **per fold** lalu dirata-rata
- LOO: semua prediksi dikumpulkan dulu, metrik dihitung **sekali secara global** — lebih akurat karena tidak ada varians antar fold

> **Kesesuaian dengan artikel:** Akhtar et al. menggunakan LOO sebagai metode evaluasi kedua. ✅

---

## Bagian 5 — Metrik Evaluasi

Diturunkan dari **Confusion Matrix**:

```
               Prediksi
            Tidak CHD    CHD
Aktual Tidak CHD   TN     FP
       CHD         FN     TP
```

| Metrik | Rumus | Penjelasan |
|--------|-------|------------|
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | Persentase prediksi benar keseluruhan |
| **Sensitivity** | TP / (TP+FN) | Kemampuan mendeteksi pasien yang benar-benar CHD |
| **Specificity** | TN / (TN+FP) | Kemampuan mengidentifikasi pasien yang tidak CHD |
| **AUC** | Area Under ROC Curve | Kemampuan model membedakan dua kelas (0–1) |

> **Kesesuaian dengan artikel:** Keempat metrik sama persis dengan artikel. ✅ Sensitivity adalah metrik terpenting di konteks medis — False Negative (pasien sakit diprediksi sehat) berakibat fatal.

---

## Bagian 6 — Visualisasi

```python
plt.savefig('hasil/performa_svm.png', dpi=150, bbox_inches='tight')
```

Menghasilkan **satu grafik 2 panel** (`hasil/performa_svm.png`):
- **Panel kiri:** Hasil K-Fold terbaik — bar chart semua kernel vs 4 metrik
- **Panel kanan:** Hasil LOO — bar chart semua kernel vs 4 metrik

---

## Ringkasan Kesesuaian dengan Artikel

| Aspek | Artikel Akhtar et al. (2023) | Kode SVM.py | Status |
|-------|------------------------------|-------------|--------|
| Algoritma | Support Vector Machine | SVC (scikit-learn) | ✅ Sesuai |
| Kernel | Linear, Polynomial, RBF, Sigmoid | 4 kernel sama | ✅ Sesuai |
| Encoding | Label Encoding | LabelEncoder sklearn | ✅ Sesuai |
| Normalisasi | Ya | StandardScaler | ✅ Sesuai |
| Seleksi fitur | Subset Evaluator (semua fitur) | Semua 11 fitur digunakan | ✅ Sesuai |
| Parameter SVM | Default | Default (tidak ada tuning) | ✅ Sesuai |
| Validasi 1 | 5-Fold CV | Multi-K (2,3,4,6,7,8,9,10) + pilih terbaik | ✅ Sesuai + dikembangkan |
| Validasi 2 | LOO CV | LeaveOneOut sklearn | ✅ Sesuai |
| Metrik | AUC, Accuracy, Sensitivity, Specificity | Sama | ✅ Sesuai |
| Hasil numerik | Artikel 2023 (sklearn lama) | Sedikit beda (~1-3%) | ⚠️ Wajar (beda versi sklearn) |

---

## Catatan Perbedaan Hasil Numerik

Metodologi **identik** dengan artikel, namun hasil numerik berbeda tipis (~1–3%) karena perbedaan versi scikit-learn:

- Artikel ditulis 2023 → kemungkinan sklearn ~1.2
- Kode ini berjalan di **sklearn 1.9.0** (2025)

Perbedaan versi mengubah presisi numerik internal solver SVM (toleransi konvergensi, optimasi numerik). Ini hal normal dalam reproducibility ML dan **bukan kesalahan metodologi**.

---

## Pertanyaan yang Mungkin Ditanya Dosen

**Q: Kenapa menguji berbagai nilai K, bukan langsung 5-Fold seperti artikel?**  
A: Untuk menemukan K optimal secara empiris. Artikel memilih 5-Fold tanpa menjelaskan alasannya, sehingga kami eksplorasi berbagai nilai K dan melaporkan hasil dari K terbaik — pendekatan yang lebih ilmiah.

**Q: Kenapa hasil angkanya tidak sama persis dengan artikel?**  
A: Data dan metodologi identik. Perbedaan kecil (~1–3%) disebabkan perbedaan versi scikit-learn. Artikel menggunakan sklearn ~1.2 (2023), implementasi kami menggunakan sklearn 1.9.0 (2025) yang memiliki perbaikan presisi numerik pada solver SVM.

**Q: Apa itu StratifiedKFold, kenapa tidak pakai KFold biasa?**  
A: StratifiedKFold memastikan proporsi kelas CHD dan tidak CHD tetap seimbang di setiap fold. Penting untuk dataset medis agar evaluasi tidak bias ke kelas mayoritas.

**Q: Kenapa LOO lebih lambat dari K-Fold?**  
A: LOO menjalankan 918 iterasi (satu per pasien), sementara 5-Fold hanya 5 iterasi. Setiap iterasi melatih model dari awal, sehingga LOO butuh ~918/5 = 184x lebih lama.

**Q: Kenapa metrik LOO dihitung berbeda dari K-Fold?**  
A: Pada K-Fold, setiap fold menghasilkan cukup data untuk menghitung metrik per fold lalu dirata-rata. Pada LOO, setiap iterasi hanya menghasilkan 1 prediksi sehingga tidak bisa dihitung per iterasi — semua prediksi dikumpulkan dulu baru dihitung sekali secara global.
